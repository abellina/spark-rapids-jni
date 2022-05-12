/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <cwctype>
#include <iostream>
#include <unordered_set>

// TCompactProtocol requires some #defines to work right.
// This came from the parquet code itself...
#define SIGNED_RIGHT_SHIFT_IS 1
#define ARITHMETIC_RIGHT_SHIFT 1
#include <thrift/TApplicationException.h>
#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include <cudf/detail/nvtx/ranges.hpp>
#include <parquet_types.h>

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"

namespace rapids {
namespace jni {

struct schema_info {
  uint64_t schema_gather_ix;
  int schema_num_children;
};

/**
 * Convert a string to lower case. It uses std::tolower per character which has limitations
 * and may not produce the exact same result as the JVM does. This is probably good enough
 * for now.
 */
std::string unicode_to_lower(std::string const& input) {
  // get the size of the wide character result
    std::size_t wide_size = std::mbstowcs(nullptr, input.data(), 0);
  if (wide_size < 0) {
    throw std::invalid_argument("invalid character sequence");
  }

  std::vector<wchar_t> wide(wide_size + 1);
  // Set a null so we can get a proper output size from wcstombs. This is because 
  // we pass in a max length of 0, so it will only stop when it see the null character.
  wide.back() = 0;
  if (std::mbstowcs(wide.data(), input.data(), wide_size) != wide_size) {
    throw std::runtime_error("error during wide char converstion");
  }
  for (auto wit = wide.begin(); wit != wide.end(); ++wit) {
    *wit = std::towlower(*wit);
  }
  // Get the multi-byte result size
  std::size_t mb_size = std::wcstombs(nullptr, wide.data(), 0);
  if (mb_size < 0) {
    throw std::invalid_argument("unsupported wide character sequence");
  }
  // We are allocating a fixed size string so we can put the data directly into it
  // instead of going through a NUL terminated char* first. The NUL fill char is
  // just because we need to pass in a fill char. The value does not matter
  // because it will be overwritten. std::string itself will insert a NUL
  // terminator on the buffer it allocates internally. We don't need to worry about it.
  std::string ret(mb_size, '\0');
  if (std::wcstombs(ret.data(), wide.data(), mb_size) != mb_size) {
    throw std::runtime_error("error during multibyte char converstion");
  }
  return ret;
}

/**
 * Holds a set of "maps" that are used to rewrite various parts of the parquet metadata.
 * Generally each "map" is a gather map that pulls data from an input vector to be placed in
 * an output vector.
 */
struct column_pruning_maps {
  // gather map for pulling out items from the schema
  // Each SchemaElement also includes the number of children in it. This allows the vector
  // to be interpreted as a tree flattened depth first. These are the new values for num
  // children after the schema is gathered.
  std::vector<schema_info> schema_map;

  // There are several places where a struct is stored only for a leaf column (like a column chunk)
  // This holds the gather map for those cases.
  std::vector<int> chunk_map;
};


static bool invalid_file_offset(long start_index, long pre_start_index, long pre_compressed_size) {
  bool invalid = false;
  // checking the first rowGroup
  if (pre_start_index == 0 && start_index != 4) {
    invalid = true;
    return invalid;
  }

  //calculate start index for other blocks
  int64_t min_start_index = pre_start_index + pre_compressed_size;
  if (start_index < min_start_index) {
    // a bad offset detected, try first column's offset
    // can not use minStartIndex in case of padding
    invalid = true;
  }

  return invalid;
}

static int64_t get_offset(rapids::parquet::format::ColumnChunk const& column_chunk) {
  auto md = column_chunk.meta_data;
  int64_t offset = md.data_page_offset;
  if (md.__isset.dictionary_page_offset && offset > md.dictionary_page_offset) {
    offset = md.dictionary_page_offset;
  }
  return offset;
}
/**
 * This class will handle processing column pruning for a schema. It is written as a class because
 * of JNI we are sending the names of the columns as a depth first list, like parquet does internally.
 */
class column_pruner {
public:
    /**
     * Create pruning filter from a depth first flattened tree of names and num_children.
     * The root entry is not included in names or in num_children, but parent_num_children
     * should hold how many entries there are in it.
     */
    column_pruner(const std::vector<std::string> & names, 
            const std::vector<int> & num_children, 
            int parent_num_children,
            int64_t part_offset,
            int64_t part_length): children(), s_id(0), c_id(-1),
            _part_offset(part_offset),
            _part_length(part_length) {
      add_depth_first(names, num_children, parent_num_children);
      init();
    }

    column_pruner(int s_id, int c_id): children(), s_id(s_id), c_id(c_id),
      _part_offset(0), _part_length(0) {
      init();
    }

    column_pruner(): children(), s_id(0), c_id(-1),
      _part_offset(0), _part_length(0) {
      init();
    }

    std::map<int, int, std::less<int>> chunk_map;
    std::unordered_set<int> interesting_chunks;
    std::map<int, schema_info, std::less<int>> schema_map;
    std::vector<int> num_children_stack;
    std::vector<column_pruner*> tree_stack;
    std::vector<rapids::parquet::format::SchemaElement> schema_items;
    uint64_t chunk_index;
    uint64_t schema_index;

    // This is based off of the java parquet_mr code to find the groups in a range...
    //auto num_row_groups = meta.row_groups.size();
    int64_t pre_start_index;
    int64_t pre_compressed_size;

    bool first_column_with_metadata;
    uint64_t row_groups_so_far;
    std::vector<rapids::parquet::format::RowGroup> filtered_groups;
    std::vector<rapids::parquet::format::ColumnOrder> column_orders;
    std::vector<rapids::parquet::format::KeyValue> key_values;

    void init() {
      CUDF_FUNC_RANGE();

      chunk_index = 0;
      schema_index = 0;

      pre_start_index = 0;
      pre_compressed_size = 0;
      first_column_with_metadata = true;
      row_groups_so_far = 0;
    }

    /**
     * Given a schema from a parquet file create a set of pruning maps to prune columns from the rest of the footer
     */
    void filter_schema(const rapids::parquet::format::SchemaElement& schema_item, bool ignore_case) {
      CUDF_FUNC_RANGE();

      // TODO: remove this unnecesary copy
      schema_items.push_back(schema_item);
      // We are skipping over the first entry in the schema because it is always the root entry, and
      //  we already processed it
      if (schema_index == 0) {
        // Start off with 0 children in the root, will add more as we go
        schema_map[0] = {0, 0};

        // num_children_stack and tree_stack hold the current state as we walk though schema
        tree_stack.push_back(this);

        num_children_stack.push_back(schema_item.num_children);
        ++schema_index;
        return;
      }

      // num_children is optional, but is supposed to be set for non-leaf nodes. That said leaf nodes
      // will have 0 children so we can just default to that.
      int num_children = 0;
      if (schema_item.__isset.num_children) {
        num_children = schema_item.num_children;
      }

      nvtxRangePush("to_lower");
      std::string name;
      if (ignore_case) {
        name = unicode_to_lower(schema_item.name);
      } else {
        name = schema_item.name;
      }

      nvtxRangePop();

      column_pruner * found = nullptr;
      nvtxRangePush("find column_pruner");
      if (tree_stack.back() != nullptr) {
        // tree_stack can have a nullptr in it if the schema we are looking through
        // has an entry that does not match the tree
        auto found_it = tree_stack.back()->children.find(name);
        if (found_it != tree_stack.back()->children.end()) {
          found = &(found_it->second);
          int parent_mapped_schema_index = tree_stack.back()->s_id;
          ++(schema_map[parent_mapped_schema_index].schema_num_children);

          int mapped_schema_index = found->s_id;
          schema_map[mapped_schema_index] = {schema_index, 0};
        }
      }
      nvtxRangePop();


      if (schema_item.__isset.type) {
        // this is a leaf node, it has a primitive type.
        if (found != nullptr) {
          int mapped_chunk_index = found->c_id;
          chunk_map[mapped_chunk_index] = chunk_index;
          interesting_chunks.insert(chunk_index);
        }
        ++chunk_index;
      }
      // else it is a non-leaf node it is group typed
      // chunks are only for leaf nodes

      // num_children and if the type is set or not should correspond to each other.
      //  By convention in parquet they should, but to be on the safe side I keep them
      //  separate.
      nvtxRangePush("end");
      if (num_children > 0) {
        tree_stack.push_back(found);
        num_children_stack.push_back(num_children);
      } else {
        // go back up the stack/tree removing children until we hit one with more children
        bool done = false;
        while (!done) {
          int parent_children_left = num_children_stack.back() - 1;
          if (parent_children_left > 0) {
            num_children_stack.back() = parent_children_left;
            done = true;
          } else {
            tree_stack.pop_back();
            num_children_stack.pop_back();
          }

          if (tree_stack.size() == 0) {
            done = true;
          }
        }
      }
      nvtxRangePop();
      ++schema_index;
    }

    void on_column_order(const rapids::parquet::format::ColumnOrder& co) {
      CUDF_FUNC_RANGE();
      column_orders.push_back(co);
    }

    void on_key_value(const rapids::parquet::format::KeyValue& kv) {
      CUDF_FUNC_RANGE();
      key_values.push_back(kv);
    }

    void filter_groups(const rapids::parquet::format::RowGroup& row_group) {
     CUDF_FUNC_RANGE();
     if (_part_length <= 0) {
       filtered_groups.push_back(row_group);
       return;
     }
     if (row_groups_so_far++ == 0) {
       first_column_with_metadata = row_group.columns[0].__isset.meta_data;
     }
     int64_t total_size = 0;
     int64_t start_index;
     auto column_chunk = row_group.columns[0];
     if (first_column_with_metadata) {
       start_index = get_offset(column_chunk);
     } else {
       //the file_offset of first block always holds the truth, while other blocks don't :
       //see PARQUET-2078 for details
       start_index = row_group.file_offset;
       if (invalid_file_offset(start_index, pre_start_index, pre_compressed_size)) {
         //first row group's offset is always 4
         if (pre_start_index == 0) {
           start_index = 4;
         } else {
           // use minStartIndex(imprecise in case of padding, but good enough for filtering)
           start_index = pre_start_index + pre_compressed_size;
         }
       }
       pre_start_index = start_index;
       pre_compressed_size = row_group.total_compressed_size;
     }
     if (row_group.__isset.total_compressed_size) {
        total_size = row_group.total_compressed_size;
     } else {
        auto num_columns = row_group.columns.size();
        for (uint64_t cc_i = 0; cc_i < num_columns; ++cc_i) {
            rapids::parquet::format::ColumnChunk const& col = row_group.columns[cc_i];
            total_size += col.meta_data.total_compressed_size;
        }
     }
     int64_t mid_point = start_index + total_size / 2;
     if (mid_point >= _part_offset && mid_point < (_part_offset + _part_length)) {
       filtered_groups.push_back(row_group);
     }
    }

    column_pruning_maps get_maps() {
      // If there is a column that is missing from this file we need to compress the gather maps
      //  so there are no gaps
      std::vector<schema_info> final_schema_map;
      final_schema_map.reserve(schema_map.size());
      for (auto it = schema_map.begin(); it != schema_map.end(); ++it) {
        final_schema_map.push_back(it->second);
      }

      std::vector<int> final_chunk_map;
      final_chunk_map.reserve(chunk_map.size());
      for (auto it = chunk_map.begin(); it != chunk_map.end(); ++it) {
        final_chunk_map.push_back(it->second);
      }

      return column_pruning_maps{std::move(final_schema_map),
          std::move(final_chunk_map)};
    }

private:

    void add_depth_first(std::vector<std::string> const& names,
            std::vector<int> const& num_children,
            int parent_num_children) {
      CUDF_FUNC_RANGE();
      if (parent_num_children == 0) {
        // There is no point in doing more the tree is empty, and it lets us avoid some corner cases
        // in the code below
        return;
      }
      int local_s_id = 0; // There is always a root on the schema
      int local_c_id = -1; // for columns it is just the leaf nodes
      auto num = names.size();
      std::vector<column_pruner*> tree_stack;
      std::vector<int> num_children_stack;
      tree_stack.push_back(this);
      num_children_stack.push_back(parent_num_children);
      for(uint64_t i = 0; i < num; ++i) {
        auto name = names[i];
        auto num_c = num_children[i];
        ++local_s_id;
        int tmp_c_id = -1;
        if (num_c == 0) {
          // leaf node...
          ++local_c_id;
          tmp_c_id = local_c_id;
        }
        tree_stack.back()->children.try_emplace(name, local_s_id, tmp_c_id);
        if (num_c > 0) {
          tree_stack.push_back(&tree_stack.back()->children[name]);
          num_children_stack.push_back(num_c);
        } else {
          // go back up the stack/tree removing children until we hit one with more children
          bool done = false;
          while (!done) {
              int parent_children_left = num_children_stack.back() - 1;
              if (parent_children_left > 0) {
                num_children_stack.back() = parent_children_left;
                done = true;
              } else {
                tree_stack.pop_back();
                num_children_stack.pop_back();
              }

              if (tree_stack.size() <= 0) {
                done = true;
              }
          }
        }
      }
      if (tree_stack.size() != 0 || num_children_stack.size() != 0) {
        throw std::invalid_argument("DIDN'T CONSUME EVERYTHING...");
      }
    }

    std::map<std::string, column_pruner> children;
    // The following IDs are the position that they should be in when output in a filtered footer, except
    // that if there are any missing columns in the actual data the gaps need to be removed.
    // schema ID
    int s_id;
    // Column chunk and Column order ID
    int c_id;

    int64_t _part_offset;
    int64_t _part_length;
};



using ThriftBuffer = apache::thrift::transport::TMemoryBuffer;
using ThriftProtocol = apache::thrift::protocol::TProtocol;

class transport_protocol : public rapids::parquet::format::FileMetaDataListener {
public:
  virtual void on_schema(const rapids::parquet::format::SchemaElement& se){
    _pruner->filter_schema(se, _ignore_case);
  }
  virtual void on_key_value(const rapids::parquet::format::KeyValue& kv){
    _pruner->on_key_value(kv);
  }
  virtual void on_row_group(const rapids::parquet::format::RowGroup& rg){
    _pruner->filter_groups(rg);
  }
  virtual bool on_row_group_column(uint32_t ix) {
    bool should_skip =
      _pruner->interesting_chunks.find(ix) == _pruner->interesting_chunks.end();
    nvtxRangePush(should_skip ? "skip" : "keep");
    nvtxRangePop();
    return !should_skip;
  }
  // NOT called right now
  virtual void on_column_order(const rapids::parquet::format::ColumnOrder& co){
    _pruner->on_column_order(co);
  }
  virtual void on_created_by() {
    CUDF_FUNC_RANGE();
  }
  virtual void on_encryption_algo() {
    CUDF_FUNC_RANGE();
  }
  virtual void on_signing_key_meta() {
    CUDF_FUNC_RANGE();
  }
  virtual void on_just_skip() {
    CUDF_FUNC_RANGE();
  }

  //virtual void on_start(const char* msg){
  //  nvtxRangePush(msg);
  //}
  //virtual void on_end(){
  //  nvtxRangePop();
  //}
  virtual ~transport_protocol() {
    t_transport.reset();
    t_protocol.reset();
  }

  std::shared_ptr<ThriftBuffer> t_transport;
  std::shared_ptr<ThriftProtocol> t_protocol;

  void reset_transport(uint8_t* buffer, uint32_t len) {
    t_transport->resetBuffer(buffer, len);
  }

  column_pruner* _pruner;
  bool _ignore_case;

  void set_pruner(column_pruner* p, bool ignore_case) {
    _pruner = p;
    _ignore_case = ignore_case;
  }
};

transport_protocol* initialize() {
  // A lot of this came from the parquet source code...
  // Deserialize msg bytes into c++ thrift msg using memory transport.
  #if PARQUET_THRIFT_VERSION_MAJOR > 0 || PARQUET_THRIFT_VERSION_MINOR >= 14
  auto conf = std::make_shared<apache::thrift::TConfiguration>();
  conf->setMaxMessageSize(std::numeric_limits<int>::max());
  auto tmem_transport = new ThriftBuffer(
    nullptr,
    0,
    ThriftBuffer::OBSERVE,
    conf);
  #else
  auto tmem_transport = new ThriftBuffer(
    nullptr,
    0);
  #endif

  auto tp = new transport_protocol();
  tp->t_transport.reset(tmem_transport);

  apache::thrift::protocol::TCompactProtocolFactoryT<ThriftBuffer> tproto_factory;
  // Protect against CPU and memory bombs
  tproto_factory.setStringSizeLimit(100 * 1000 * 1000);
  // Structs in the thrift definition are relatively large (at least 300 bytes).
  // This limits total memory to the same order of magnitude as stringSize.
  tproto_factory.setContainerSizeLimit(1000 * 1000);
  tp->t_protocol = tproto_factory.getProtocol(tp->t_transport);//, tp);

  return tp;
}

void deserialize_parquet_footer(transport_protocol* tp, rapids::parquet::format::FileMetaData * meta) {
  CUDF_FUNC_RANGE();
  try {
    meta->read(tp->t_protocol.get());
  } catch (std::exception& e) {
    std::stringstream ss;
    ss << "Couldn't deserialize thrift: " << e.what() << "\n";
    throw std::runtime_error(ss.str());
  }
}

void deserialize_parquet_footer(uint8_t * buffer, uint32_t len, rapids::parquet::format::FileMetaData * meta) {
  CUDF_FUNC_RANGE();

  auto tp = initialize();
  tp->reset_transport(buffer, len);

  try {
    meta->read(tp->t_protocol.get());
    delete tp;
  } catch (std::exception& e) {
    delete tp;
    std::stringstream ss;
    ss << "Couldn't deserialize thrift: " << e.what() << "\n";
    throw std::runtime_error(ss.str());
  }
}

void filter_columns(std::vector<rapids::parquet::format::RowGroup> & groups, std::vector<int> & chunk_filter) {
  CUDF_FUNC_RANGE();
  for (auto group_it = groups.begin(); group_it != groups.end(); ++group_it) {
    std::vector<rapids::parquet::format::ColumnChunk> new_chunks;
    for (auto it = chunk_filter.begin(); it != chunk_filter.end(); ++it) {
      new_chunks.push_back(group_it->columns[*it]);
    }
    group_it->columns = std::move(new_chunks);
  }
}

}
}

extern "C" {
JNIEXPORT long JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_initialize(JNIEnv * env, jclass) {
  auto tp = rapids::jni::initialize();
  return (long)tp;
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_reset(JNIEnv * env, jclass,
    jlong tpaddr, jlong buffer, jlong buffer_length) {
  auto tp = reinterpret_cast<rapids::jni::transport_protocol*>(tpaddr);
  tp->reset_transport(reinterpret_cast<uint8_t*>(buffer), buffer_length);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_delete(JNIEnv * env, jclass,
    jlong tpaddr) {
  auto tp = reinterpret_cast<rapids::jni::transport_protocol*>(tpaddr);
  delete tp;
}

JNIEXPORT long JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_readAndFilter2(JNIEnv * env, jclass,
                                                                                    jlong tpaddr,
                                                                                    jlong part_offset,
                                                                                    jlong part_length,
                                                                                    jobjectArray filter_col_names,
                                                                                    jintArray num_children,
                                                                                    jint parent_num_children,
                                                                                    jboolean ignore_case) {
  CUDF_FUNC_RANGE();
  try {
    auto tp = reinterpret_cast<rapids::jni::transport_protocol*>(tpaddr);

    // special meta that can takes a FileMetaDataListener
    auto meta = std::make_unique<rapids::parquet::format::FileMetaData>(tp);

    // Get the filter for the columns first...
    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
    cudf::jni::native_jintArray n_num_children(env, num_children);

    rapids::jni::column_pruner pruner(n_filter_col_names.as_cpp_vector(),
            std::vector(n_num_children.begin(), n_num_children.end()),
            parent_num_children,
            part_offset, part_length);

    // set the pruner in our listener
    tp->set_pruner(&pruner, ignore_case);

    // this calls read() and all the callbacks
    rapids::jni::deserialize_parquet_footer(tp, meta.get());

    auto filter = pruner.get_maps();

    nvtxRangePush("filter schema");
    // start by filtering the schema and the chunks
    std::size_t new_schema_size = filter.schema_map.size();
    std::vector<rapids::parquet::format::SchemaElement> new_schema(new_schema_size);
    for (std::size_t i = 0; i < new_schema_size; ++i) {
      int orig_index = filter.schema_map[i].schema_gather_ix;
      int new_num_children = filter.schema_map[i].schema_num_children;
      new_schema[i] = pruner.schema_items[orig_index];
      new_schema[i].num_children = new_num_children;
    }
    meta->schema = std::move(new_schema);
    nvtxRangePop();

    nvtxRangePush("set column orders");
    if (pruner.column_orders.size() > 0) {
      std::vector<rapids::parquet::format::ColumnOrder> new_order;
      for (auto it = filter.chunk_map.begin(); it != filter.chunk_map.end(); ++it) {
        new_order.push_back(pruner.column_orders[*it]);
      }
      meta->__isset.column_orders = true;
      meta->column_orders = std::move(new_order);
    }
    nvtxRangePop();

    nvtxRangePush("set row_groups");
    // Now we want to filter the columns out of each row group that we care about as we go.
    meta->row_groups = std::move(pruner.filtered_groups);
    meta->key_value_metadata = std::move(pruner.key_values);
    nvtxRangePop();

    rapids::jni::filter_columns(meta->row_groups, filter.chunk_map);

    return cudf::jni::release_as_jlong(meta);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT long JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_readAndFilter(JNIEnv * env, jclass,
                                                                                    jlong part_offset,
                                                                                    jlong part_length,
                                                                                    jobjectArray filter_col_names,
                                                                                    jintArray num_children,
                                                                                    jint parent_num_children,
                                                                                    jboolean ignore_case) {
  return 0;
  //CUDF_FUNC_RANGE();
  //try {
  //  auto tp = reinterpret_cast<rapids::jni::transport_protocol*>(tpaddr);
  //  auto meta = std::make_unique<rapids::parquet::format::FileMetaData>(tp);
  //  uint32_t len = static_cast<uint32_t>(buffer_length);

  //  // We don't support encrypted parquet...
  //  rapids::jni::deserialize_parquet_footer(tp, meta.get());

  //  // Get the filter for the columns first...
  //  cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
  //  cudf::jni::native_jintArray n_num_children(env, num_children);

  //  rapids::jni::column_pruner pruner(n_filter_col_names.as_cpp_vector(),
  //          std::vector(n_num_children.begin(), n_num_children.end()),
  //          parent_num_children);

  //  // call from on_schema
  //  pruner.filter_schema(meta->schema, ignore_case);

  //  // maybe once on_schema is done this happens?
  //  auto filter = pruner.get_maps();

  //  // start by filtering the schema and the chunks
  //  std::size_t new_schema_size = filter.schema_map.size();
  //  std::vector<rapids::parquet::format::SchemaElement> new_schema(new_schema_size);
  //  for (std::size_t i = 0; i < new_schema_size; ++i) {
  //    int orig_index = filter.schema_map[i];
  //    int new_num_children = filter.schema_num_children[i];
  //    new_schema[i] = pruner.schema_items[orig_index];
  //    new_schema[i].num_children = new_num_children;
  //  }
  //  meta->schema = std::move(new_schema);
  //  if (meta->__isset.column_orders) {
  //    std::vector<rapids::parquet::format::ColumnOrder> new_order;
  //    for (auto it = filter.chunk_map.begin(); it != filter.chunk_map.end(); ++it) {
  //      new_order.push_back(meta->column_orders[*it]);
  //    }
  //    meta->column_orders = std::move(new_order);
  //  }
  //  // Now we want to filter the columns out of each row group that we care about as we go.
  //  if (part_length >= 0) {
  //    meta->row_groups = std::move(rapids::jni::filter_groups(*meta, part_offset, part_length));
  //  }
  //  rapids::jni::filter_columns(meta->row_groups, filter.chunk_map);

  //  return cudf::jni::release_as_jlong(meta);
  //}
  //CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_close(JNIEnv * env, jclass,
                                                                            jlong handle) {
  try {
    rapids::parquet::format::FileMetaData * ptr = reinterpret_cast<rapids::parquet::format::FileMetaData *>(handle);
    delete ptr;
  }
  CATCH_STD(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_getNumRows(JNIEnv * env, jclass,
                                                                                 jlong handle) {
  try {
    rapids::parquet::format::FileMetaData * ptr = reinterpret_cast<rapids::parquet::format::FileMetaData *>(handle);
    long ret = 0;
    for(auto it = ptr->row_groups.begin(); it != ptr->row_groups.end(); ++it) {
      ret = ret + it->num_rows;
    }
    return ret;
  }
  CATCH_STD(env, -1);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_getNumColumns(JNIEnv * env, jclass,
                                                                                     jlong handle) {
  try {
    rapids::parquet::format::FileMetaData * ptr = reinterpret_cast<rapids::parquet::format::FileMetaData *>(handle);
    int ret = 0;
    if (ptr->schema.size() > 0) {
      if (ptr->schema[0].__isset.num_children) {
        ret = ptr->schema[0].num_children;
      }
    }
    return ret;
  }
  CATCH_STD(env, -1);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_ParquetFooter_serializeThriftFile(JNIEnv * env, jclass,
                                                                                             jlong handle) {
  CUDF_FUNC_RANGE();
  try {
    rapids::parquet::format::FileMetaData * meta = reinterpret_cast<rapids::parquet::format::FileMetaData *>(handle);
    std::shared_ptr<apache::thrift::transport::TMemoryBuffer> transportOut(
            new apache::thrift::transport::TMemoryBuffer());
    apache::thrift::protocol::TCompactProtocolFactoryT<apache::thrift::transport::TMemoryBuffer> factory;
    auto protocolOut = factory.getProtocol(transportOut);
    meta->write(protocolOut.get());
    uint8_t * buf_ptr;
    uint32_t buf_size;
    transportOut->getBuffer(&buf_ptr, &buf_size);

    // 12 extra is for the MAGIC thrift_footer length MAGIC
    jobject ret = cudf::jni::allocate_host_buffer(env, buf_size + 12, false);
    uint8_t* ret_addr = reinterpret_cast<uint8_t*>(cudf::jni::get_host_buffer_address(env, ret));
    ret_addr[0] = 'P';
    ret_addr[1] = 'A';
    ret_addr[2] = 'R';
    ret_addr[3] = '1';
    std::memcpy(ret_addr + 4, buf_ptr, buf_size);
    uint8_t * after = ret_addr + buf_size + 4;
    after[0] = static_cast<uint8_t>(0xFF & buf_size);
    after[1] = static_cast<uint8_t>(0xFF & (buf_size >> 8));
    after[2] = static_cast<uint8_t>(0xFF & (buf_size >> 16));
    after[3] = static_cast<uint8_t>(0xFF & (buf_size >> 24));
    after[4] = 'P';
    after[5] = 'A';
    after[6] = 'R';
    after[7] = '1';
    return ret;
  }
  CATCH_STD(env, nullptr);
}

}

/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>

#include <cudf/io/parquet.hpp>

#include <parse_uri.hpp>

struct ReadParquetTests : public cudf::test::BaseFixture {};

cudf::io::table_with_metadata read_parquet(
  std::string const& file_path)
{
  std::cout << "reading parquet file: " << file_path << std::endl;
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::parquet_reader_options::builder(source_info);
  cudf::io::parquet_reader_options options = builder.build();
  return cudf::io::read_parquet(options);
}

TEST_F(ReadParquetTests, ReadParquet)
{
  char * parquet_file_name = getenv("T_PARQUET_FILE");
  if (parquet_file_name == nullptr) {
    FAIL() << "please specify a file using T_PARQUET_FILE environment variable" << std::endl;
  }

  read_parquet(std::string(parquet_file_name));
}
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "matrix.h"
#include "test.h"

namespace nda {

TEST(matrix_slice) {
  matrix<float> m({10, 10}, 0);
  for_all_indices(m.shape(), [&](int i, int j) { m(i, j) = i * 10 + j; });

  for (int i = 0; i < 10; i++) {
    auto row = m(_, i);
    auto col = m(i, _);

    for (int j = 0; j < 10; j++) {
      ASSERT_EQ(row(j), j * 10 + i);
      ASSERT_EQ(col(j), i * 10 + j);
    }
  }
}

TEST(matrix_transpose) {
  matrix<int> m({5, 8});
  fill_pattern(m);
  auto mt = transpose<1, 0>(m);
  for_all_indices(m.shape(), [&](int i, int j) { ASSERT_EQ(m(i, j), mt(j, i)); });
}

TEST(small_matrix) {
  using matrix4x3i = small_matrix<int, 4, 3>;

  matrix4x3i auto_array;
  for_all_indices(auto_array.shape(), [&](int x, int y) { auto_array(x, y) = x; });

  matrix4x3i copy_array(auto_array);
  for_all_indices(copy_array.shape(), [&](int x, int y) { ASSERT_EQ(copy_array(x, y), x); });

  matrix4x3i assign_array;
  assign_array = auto_array;
  for_all_indices(assign_array.shape(), [&](int x, int y) { ASSERT_EQ(assign_array(x, y), x); });

  matrix4x3i move_array(std::move(auto_array));
  for_all_indices(move_array.shape(), [&](int x, int y) { ASSERT_EQ(move_array(x, y), x); });

  matrix4x3i move_assign;
  move_assign = std::move(assign_array);
  for_all_indices(move_assign.shape(), [&](int x, int y) { ASSERT_EQ(move_assign(x, y), x); });
}

} // namespace nda

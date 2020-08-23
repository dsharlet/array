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

#include "array.h"
#include "test.h"

namespace nda {

typedef dense_array<int, 3, auto_allocator<int, 32>> dense3d_int_auto_array;

// Check that the data array of a is part of a itself.
template <typename T>
bool is_auto_allocation(const T& a) {
  const uint8_t* a_begin = reinterpret_cast<const uint8_t*>(&a);
  const uint8_t* a_end = a_begin + sizeof(T);
  const uint8_t* a_data_begin = reinterpret_cast<const uint8_t*>(a.data());
  const uint8_t* a_data_end = reinterpret_cast<const uint8_t*>(a.data() + a.shape().flat_extent());
  return a_begin <= a_data_begin && a_data_end <= a_end;
}

TEST(auto_array) {
  dense3d_int_auto_array auto_array({4, 3, 2});
  ASSERT(is_auto_allocation(auto_array));
  for_all_indices(auto_array.shape(), [&](int x, int y, int c) { auto_array(x, y, c) = x; });

  dense3d_int_auto_array copy_array(auto_array);
  ASSERT(is_auto_allocation(copy_array));
  for_all_indices(
      copy_array.shape(), [&](int x, int y, int c) { ASSERT_EQ(copy_array(x, y, c), x); });

  dense3d_int_auto_array assign_array;
  assign_array = auto_array;
  ASSERT(is_auto_allocation(assign_array));
  for_all_indices(
      assign_array.shape(), [&](int x, int y, int c) { ASSERT_EQ(assign_array(x, y, c), x); });

  dense3d_int_auto_array move_array(std::move(auto_array));
  ASSERT(is_auto_allocation(move_array));
  for_all_indices(
      move_array.shape(), [&](int x, int y, int c) { ASSERT_EQ(move_array(x, y, c), x); });

  dense3d_int_auto_array move_assign;
  move_assign = std::move(assign_array);
  ASSERT(is_auto_allocation(move_assign));
  for_all_indices(
      move_assign.shape(), [&](int x, int y, int c) { ASSERT_EQ(move_assign(x, y, c), x); });
}

TEST(auto_array_bad_alloc) {
  // This array is too big for our auto allocator.
  dense3d_int_auto_array not_auto_array({4, 3, 5});
  ASSERT(!is_auto_allocation(not_auto_array));
}

} // namespace nda

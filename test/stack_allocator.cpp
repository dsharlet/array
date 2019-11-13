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

typedef dense_array<int, 3, stack_allocator<int, 32>> dense3d_int_stack_array;

TEST(stack_array) {
  dense3d_int_stack_array stack_array({4, 3, 2});
  for_all_indices(stack_array.shape(), [&](int x, int y, int c) {
    stack_array(x, y, c) = x;
  });

  dense3d_int_stack_array copy_array(stack_array);
  for_all_indices(copy_array.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(copy_array(x, y, c), x);
  });

  dense3d_int_stack_array assign_array;
  assign_array = stack_array;
  for_all_indices(assign_array.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(assign_array(x, y, c), x);
  });

  dense3d_int_stack_array move_array(std::move(stack_array));
  for_all_indices(move_array.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(move_array(x, y, c), x);
  });

  dense3d_int_stack_array move_assign;
  move_assign = std::move(assign_array);
  for_all_indices(move_assign.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(move_assign(x, y, c), x);
  });
}

TEST(stack_array_bad_alloc) {
  try {
    // This array is too big for our stack allocator.
    dense3d_int_stack_array stack_array({4, 3, 5});
    ASSERT(false);
  } catch (const std::bad_alloc&) {
    // This is success.
  }
}

}  // namespace nda

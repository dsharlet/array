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

#include <cstring>

namespace nda {

TEST(array_ref_indices) {
  int data[100];
  for (int i = 0; i < 100; i++) {
    data[i] = i;
  }

  dense_array_ref<int, 1> ref_1d(data, make_dense_shape(100));
  for_all_indices(ref_1d.shape(), [&](int x) {
    ASSERT_EQ(ref_1d(x), x);
  });

  dense_array_ref<int, 2> ref_2d(data, make_dense_shape(20, 5));
  ASSERT_EQ(ref_2d.width(), 20);
  ASSERT_EQ(ref_2d.height(), 5);
  ASSERT_EQ(ref_2d.rows(), 20);
  ASSERT_EQ(ref_2d.columns(), 5);
  for_all_indices(ref_2d.shape(), [&](int x, int y) {
    ASSERT_EQ(ref_2d(x, y), y*20 + x);
  });
}

TEST(reinterpret) {
  float eight = 8.0f;
  int eight_int;
  ASSERT_EQ(sizeof(eight), sizeof(eight_int));
  std::memcpy(&eight_int, &eight, sizeof(eight));

  dense_array<int, 3> int_array({4, 5, 6}, eight_int);
  dense_array_ref<float, 3> float_array = reinterpret<float>(int_array);
  ASSERT_EQ(float_array.width(), 4);
  ASSERT_EQ(float_array.height(), 5);
  ASSERT_EQ(float_array.channels(), 6);
  ASSERT_EQ(float_array.rows(), 4);
  ASSERT_EQ(float_array.columns(), 5);
  for_all_indices(int_array.shape(), [&](int x, int y, int z) {
    ASSERT_EQ(int_array(x, y, z), eight_int);
    ASSERT_EQ(float_array(x, y, z), eight);
  });
}

TEST(array_ref_copy) {
  int data[100];
  for (int i = 0; i < 100; i++) {
    data[i] = i;
  }

  array_ref_of_rank<int, 1> evens(data, make_shape(dim<>(0, 50, 2)));
  dense_array<int, 1> evens_copy = make_dense_copy(evens);
  for (int i = 0; i < 50; i++) {
    ASSERT_EQ(evens(i), i * 2);
    ASSERT_EQ(evens_copy(i), i * 2);
  }
}

TEST(array_ref_empty) {
  dense_array_ref<int, 1> null_ref(nullptr, make_dense_shape(10));
  ASSERT(null_ref.empty());
  null_ref.set_shape({{3, 3}}, 3);
  ASSERT(null_ref.empty());
}

}  // namespace nda

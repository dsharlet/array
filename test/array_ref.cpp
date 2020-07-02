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

  dense_array_ref<int, 1> ref_1d(data, {100});
  for_all_indices(ref_1d.shape(), [&](int x) {
    ASSERT_EQ(ref_1d(x), x);
  });

  dense_array_ref<int, 2> ref_2d(data, {20, 5});
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

  array_ref_of_rank<int, 1> evens(data, {dim<>(0, 50, 2)});
  dense_array<int, 1> evens_copy = make_dense_copy(evens);
  for (int i = 0; i < 50; i++) {
    ASSERT_EQ(evens(i), i * 2);
    ASSERT_EQ(evens_copy(i), i * 2);
  }
}

TEST(array_ref_empty) {
  dense_array_ref<int, 1> null_ref(nullptr, {10});
  ASSERT(null_ref.empty());
  null_ref.set_shape({{3, 3}}, 3);
  ASSERT(null_ref.empty());
}

void f_dense(const dense_array_ref<int, 3>& r) {}
void f_const_dense(const dense_array_ref<const int, 3>& r) {}

TEST(array_ref_conversion) {
  // The correctness of shape conversion is already tested elsewhere, we just
  // want to make sure this compiles here.
  array_ref_of_rank<int, 3> null_ref(nullptr, {10, 20, 30});
  dense_array_ref<int, 3> dense_null_ref(null_ref);
  null_ref = dense_null_ref;
  array_ref_of_rank<const int, 3> const_null_ref(dense_null_ref);

  // Test conversion from array_ref<T*> to array_ref<const T*>.
  f_dense(null_ref);
  f_const_dense(null_ref);
  f_const_dense(const_null_ref);

  array_of_rank<int, 3> a({5, 10, 20});
  f_dense(a);
  f_const_dense(a);

  const array_of_rank<int, 3>& ar = a;
  f_const_dense(ar);
}

TEST(array_ref_crop) {
  dense_array<int, 2> a({8, 9});
  fill_pattern(a);

  auto a_crop1_slice2 = a(range<4, 3>(), 5);
  ASSERT_EQ(a_crop1_slice2.rank(), 1);
  ASSERT_EQ(a_crop1_slice2.x().min(), 4);
  ASSERT_EQ(a_crop1_slice2.x().extent(), 3);
  // TODO: This doesn't pass because check_pattern doesn't understand
  // that there is a second dimension to the pattern.
  //check_pattern(a_crop1_slice2);

  auto a_crop1_crop2 = a(range<>{2, 6}, range<3, 4>());
  ASSERT_EQ(a_crop1_crop2.rank(), 2);
  ASSERT_EQ(a_crop1_crop2.x().min(), 2);
  ASSERT_EQ(a_crop1_crop2.x().extent(), 6);
  ASSERT_EQ(a_crop1_crop2.y().min(), 3);
  ASSERT_EQ(a_crop1_crop2.y().extent(), 4);
  check_pattern(a_crop1_crop2);

  auto a_all1_crop2 = a(_, range<3, 4>());
  ASSERT_EQ(a_all1_crop2.rank(), 2);
  ASSERT_EQ(a_all1_crop2.x().min(), 0);
  ASSERT_EQ(a_all1_crop2.x().extent(), 8);
  ASSERT_EQ(a_all1_crop2.y().min(), 3);
  ASSERT_EQ(a_all1_crop2.y().extent(), 4);
  check_pattern(a_all1_crop2);
}

}  // namespace nda

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
  for_all_indices(ref_1d.shape(), [&](int x) { ASSERT_EQ(ref_1d(x), x); });

  dense_array_ref<int, 2> ref_2d(data, {20, 5});
  ASSERT_EQ(ref_2d.width(), 20);
  ASSERT_EQ(ref_2d.height(), 5);
  ASSERT_EQ(ref_2d.rows(), 20);
  ASSERT_EQ(ref_2d.columns(), 5);
  for_all_indices(ref_2d.shape(), [&](int x, int y) { ASSERT_EQ(ref_2d(x, y), y * 20 + x); });
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

  array_of_rank<int, 0> scalar_int({}, eight_int);
  array_ref_of_rank<float, 0> scalar = reinterpret<float>(scalar_int);
  ASSERT_EQ(scalar_int(), eight_int);
  ASSERT_EQ(scalar(), eight);
}

TEST(array_ref_copy) {
  int data[100];
  for (int i = 0; i < 100; i++) {
    data[i] = i;
  }

  array_ref_of_rank<int, 1> evens(data, {dim<>(0, 50, 2)});
  dense_array<int, 1> evens_copy = make_compact_copy(evens);
  for (int i = 0; i < 50; i++) {
    ASSERT_EQ(evens(i), i * 2);
    ASSERT_EQ(evens_copy(i), i * 2);
  }

  array_ref_of_rank<int, 0> scalar(&data[5], {});
  array_of_rank<int, 0> scalar_copy = make_compact_copy(scalar);
  ASSERT_EQ(scalar(), scalar_copy());
}

TEST(array_ref_empty) {
  dense_array_ref<int, 1> null_ref(nullptr, {10});
  ASSERT(null_ref.empty());
  null_ref.set_shape({{3, 3}}, 3);
  ASSERT(null_ref.empty());

  int x;
  array_ref_of_rank<int, 0> scalar_ref(&x, {});
  ASSERT(!scalar_ref.empty());
  array_ref_of_rank<int, 0> null_scalar_ref(nullptr, {});
  ASSERT(null_scalar_ref.empty());
}

template <typename Shape>
void template_shape(const array_ref<int, Shape>&) {}
template <typename Shape>
void template_shape_const(const array_ref<const int, Shape>&) {}

void non_template(const array_ref_of_rank<int, 3>&) {}
void non_template_const(const array_ref_of_rank<const int, 3>&) {}

void non_template_dense(const dense_array_ref<int, 3>&) {}

enum {
  general = 0,
  dense = 1,
};

int overload_shape(const array_ref_of_rank<int, 3>&) { return general; }
int overload_shape(const dense_array_ref<int, 3>&) { return dense; }
int overload_shape_const(const array_ref_of_rank<const int, 3>&) { return general; }
int overload_shape_const(const dense_array_ref<const int, 3>&) { return dense; }

TEST(array_ref_implicit_conversion) {
  array_ref_of_rank<int, 3> null_ref(nullptr, {10, 20, 30});
  array_of_rank<int, 3> non_ref({5, 10, 20});
  dense_array<int, 3> dense_non_ref({5, 10, 20});
  const array_of_rank<int, 3>& const_non_ref = non_ref;

  // array -> ref
  array_ref_of_rank<int, 3> ref(non_ref);
  array_ref_of_rank<const int, 3> const_ref(const_non_ref);
  ref = non_ref;
  const_ref = non_ref;
  const_ref = const_non_ref;

  // non-const -> const
  array_ref_of_rank<const int, 3> const_null_ref(null_ref);
  const_null_ref = null_ref;
  const_ref = non_ref;
  non_template_const(null_ref);
  non_template_const(non_ref);

  // arbitrary -> dense
  dense_array_ref<int, 3> dense_null_ref(null_ref);
  dense_null_ref = null_ref;
  non_template_dense(null_ref);
  // non_template_dense(non_ref);

  // dense -> general
  array_ref_of_rank<int, 3> null_ref2(dense_null_ref);
  null_ref = dense_null_ref;
  non_template(dense_null_ref);
  // non_template(dense_non_ref);

  // nullptr -> ref
  non_template(nullptr);
  non_template_const(nullptr);
  non_template_dense(nullptr);

  ASSERT_EQ(overload_shape(null_ref), general);
  ASSERT_EQ(overload_shape(dense_null_ref), dense);
  ASSERT_EQ(overload_shape_const(null_ref), general);
  ASSERT_EQ(overload_shape_const(dense_null_ref), dense);
}

TEST(array_ref_explicit_conversion) {
  array_ref_of_rank<int, 3> null_ref(nullptr, {10, 20, 30});
  array_of_rank<int, 3> non_ref({5, 10, 20});
  dense_array<int, 3> dense_non_ref({5, 10, 20});

  auto null_ref_embedded = convert_shape<shape_of_rank<4>>(null_ref);
  ASSERT_EQ(null_ref_embedded.rank(), 4);
  ASSERT_EQ(null_ref_embedded.w().min(), 0);
  ASSERT_EQ(null_ref_embedded.w().extent(), 1);

  auto dense_null_ref_embedded = convert_shape<dense_shape<5>>(non_ref);
  ASSERT_EQ(dense_null_ref_embedded.rank(), 5);
  ASSERT_EQ(dense_null_ref_embedded.w().min(), 0);
  ASSERT_EQ(dense_null_ref_embedded.w().extent(), 1);
  ASSERT_EQ(dense_null_ref_embedded.template dim<4>().min(), 0);
  ASSERT_EQ(dense_null_ref_embedded.template dim<4>().extent(), 1);
}

TEST(array_ref_crop_slice) {
  array<int, shape<dense_dim<0>, dim<>>> a({8, 9});
  fill_pattern(a);

  auto a_slice1 = a(3, _);
  static_assert(a_slice1.rank() == 1, "");
  assert_dim_eq(a_slice1.x(), a.y());
  // TODO: This doesn't pass because check_pattern doesn't understand
  // that there is a second dimension to the pattern.
  // check_pattern(a_slice1);

  auto a_slice2 = a(_, 2);
  static_assert(a_slice2.rank() == 1, "");
  assert_dim_eq(a_slice2.x(), a.x());
  // check_pattern(a_slice2);

  auto a_crop1_slice2 = a(interval<4, 3>(), 5);
  static_assert(a_crop1_slice2.rank() == 1, "");
  assert_dim_eq(a_crop1_slice2.x(), dense_dim<4, 3>());
  // check_pattern(a_crop1_slice2);

  auto a_slice1_crop2 = a(6, interval<4, 3>());
  static_assert(a_slice1_crop2.rank() == 1, "");
  assert_dim_eq(a_slice1_crop2.x(), dim<4, 3>(4, 3, a.y().stride()));
  // check_pattern(a_slice1_crop2);

  auto a_crop1_crop2 = a(interval<>(2, 6), interval<3, 4>());
  static_assert(a_crop1_crop2.rank() == 2, "");
  assert_dim_eq(a_crop1_crop2.x(), dense_dim<>(2, 6));
  assert_dim_eq(a_crop1_crop2.y(), dim<3, 4>(3, 4, a.y().stride()));
  check_pattern(a_crop1_crop2);

  auto a_all1_crop2 = a(_, interval<3, 4>());
  static_assert(a_all1_crop2.rank() == 2, "");
  assert_dim_eq(a_all1_crop2.x(), a.x());
  assert_dim_eq(a_all1_crop2.y(), dim<3, 4>(3, 4, a.y().stride()));
  check_pattern(a_all1_crop2);

  auto a_temp_names = a(range(2, 6), fixed_interval<5>(3));
  static_assert(a_temp_names.rank() == 2, "");
  assert_dim_eq(a_temp_names.x(), dense_dim<>(2, 4));
  assert_dim_eq(a_temp_names.y(), fixed_dim<5>(3, 5, a.y().stride()));
  check_pattern(a_temp_names);
}

} // namespace nda

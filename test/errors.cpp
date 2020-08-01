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
#include "ein_reduce.h"

// These tests should each generate one error, with errors.cpp as the
// location of the error.

namespace nda {

shape_of_rank<2> s;
dense_array_ref<int, 2> ref;
dense_array<int, 3> a;
const dense_array<int, 3> const_a;

void range_range_bad_copy_construct() {
  fixed_interval<3> x;
  interval<0, 2> y(x);
}

void range_bad_assign() {
  fixed_interval<3> x;
  x = interval<0, 2>();
}

void dim_dim_bad_copy_construct() {
  dim<0, 1, 2> strided;
  dense_dim<> x2(strided);
}

void dim_bad_assign() {
  dense_dim<> x;
  x = dim<0, 1, 2>();
}

void shape_dim_bad_index() { auto z = s.template dim<2>(); }

// TODO: This one returns an error in array.h, but it's not too spammy.
// void shape_z_bad() {
//  auto z = s.z();
//}

void shape_shape_too_many_dims() { shape<dim<>> s2(0, 1); }

// TODO: This builds due to https://github.com/dsharlet/array/issues/20
// void shape_shape_incompatible() {
//  shape<dim<dynamic, dynamic, 4>> s2;
//  shape<dense_dim<>> s3(s2);
//}

void shape_at_too_many_indices() { s(0, 1, 2); }

void shape_at_too_few_indices() { s(0); }

void shape_transpose_not_permutation1() { transpose<0>(s); }

void shape_transpose_not_permutation2() { transpose<0, 0>(s); }

void shape_transpose_not_permutation3() { transpose<0, 2>(s); }

void shape_transpose_not_permutation4() {
  shape_of_rank<3> s3;
  // (4 + 2)*2*2 = 4! = 24, a tricky case for enable_if_permutation.
  transpose<0, 0, 4>(s4);
}

void shape_operator_eq_different_rank() {
  shape_of_rank<3> s2;
  s == s2;
}

void is_compatible_different_dims() { is_compatible<shape_of_rank<3>>(s); }

void for_each_index_indices() {
  for_each_index(s, [](int x, int y) {});
}

void for_all_indices_too_many_indices() {
  for_all_indices(s, [](int, int, int) {});
}

void for_all_indices_too_few_indices() {
  for_all_indices(s, [](int) {});
}

void for_all_indices_permute_too_many_indices() {
  for_all_indices<0>(s, [](int, int) {});
}

void for_all_indices_permute_too_few_indices() {
  for_all_indices<0, 1>(s, [](int) {});
}

void array_ref_at_too_many_indices() { ref(0, 1, 2); }

void array_ref_at_too_few_indices() { ref(0); }

void array_ref_for_each_value_bad_type() {
  ref.for_each_value([&](float& i) { i = 0; });
}

void array_ref_for_each_value_too_many_args() {
  ref.for_each_value([&](int i, int j) {});
}

void array_at_too_many_indices() { a(0, 1, 2, 3); }

void array_at_too_few_indices() { a(0, 1); }

void array_for_each_value_non_const() {
  const_a.for_each_value([&](int& i) { i = 0; });
}

void copy_different_rank() { copy(a, ref); }

void make_copy_different_rank() { auto a2 = make_copy(a, shape_of_rank<2>()); }

void make_copy_ref_different_rank() { auto a2 = make_copy(ref, shape_of_rank<3>()); }

void move_different_rank() { move(a, ref); }

void make_move_different_rank() { auto a2 = make_move(a, shape_of_rank<2>()); }

void make_move_ref_different_rank() { auto a2 = make_move(ref, shape_of_rank<3>()); }

void ein_wrong_rank() { ein<0, 1>(a); }

void ein_scalar_arithmetic() { ein<0, 1, 2>(a) + 2; }

void ein_reduce_not_assignment() { ein_reduce(ein<0, 1, 2>(a)); }

} // namespace nda

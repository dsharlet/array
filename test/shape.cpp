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
#include "matrix.h"
#include "test.h"

#include <algorithm>
#include <vector>

namespace nda {

TEST(shape_scalar) {
  shape<> s;
  ASSERT_EQ(s.flat_extent(), 1);
  ASSERT_EQ(s.size(), 1);
  ASSERT_EQ(s(), 0);
}

TEST(shape_1d) {
  for (int stride : {1, 2, 10}) {
    dim<> x(0, 10, stride);
    shape<dim<>> s(x);
    for (int i : x) {
      ASSERT_EQ(s(i), i * stride);
    }
  }
}

TEST(shape_1d_dense) {
  dense_dim<> x(0, 10);
  shape<dense_dim<>> s(x);
  for (int i : x) {
    ASSERT_EQ(s(i), i);
  }
}

TEST(shape_2d) {
  dense_dim<> x(0, 10);
  dim<> y(0, 5, x.extent());
  shape<dense_dim<>, dim<>> s(x, y);
  for (int i : y) {
    for (int j : x) {
      ASSERT_EQ(s(j, i), i * x.extent() + j);
    }
  }
}

TEST(shape_2d_negative_stride) {
  dense_dim<> x(0, 10);
  dim<> y(0, 5, -x.extent());
  shape<dense_dim<>, dim<>> s(x, y);
  index_t flat_min = s(s.min());
  index_t flat_max = flat_min;
  for (int i : y) {
    for (int j : x) {
      ASSERT_EQ(s(j, i), i * -x.extent() + j);
      flat_min = std::min(s(j, i), flat_min);
      flat_max = std::max(s(j, i), flat_max);
    }
  }
  ASSERT_EQ(s.size(), 50);
  ASSERT_EQ(s.flat_extent(), 50);
  ASSERT_EQ(s.flat_min(), flat_min);
  ASSERT_EQ(s.flat_max(), flat_max);

  shape_of_rank<3> s2(10, 5, {0, 3, -1});
  s2.resolve();
  ASSERT_EQ(s2.x().stride(), 3);
  ASSERT_EQ(s2.y().stride(), 30);
}

TEST(make_dense_shape_1d) {
  dense_shape<1> s(10);
  assert_dim_eq(s.x(), dense_dim<>(0, 10));
}

TEST(make_dense_shape_2d) {
  dense_shape<2> s(10, 5);
  s.resolve();
  auto x = s.x();
  auto y = s.y();
  assert_dim_eq(x, dense_dim<>(0, 10));
  assert_dim_eq(y, dim<>(0, 5, 10));

  ASSERT_EQ(s.width(), x.extent());
  ASSERT_EQ(s.height(), y.extent());
  ASSERT_EQ(s.rows(), x.extent());
  ASSERT_EQ(s.columns(), y.extent());
}

TEST(make_dense_shape_3d) {
  dense_shape<3> s(10, 5, 20);
  s.resolve();
  auto x = s.x();
  auto y = s.y();
  auto z = s.z();
  assert_dim_eq(x, dense_dim<>(0, 10));
  assert_dim_eq(y, dim<>(0, 5, 10));
  assert_dim_eq(z, dim<>(0, 20, 50));

  ASSERT_EQ(s.width(), x.extent());
  ASSERT_EQ(s.height(), y.extent());
  ASSERT_EQ(s.channels(), z.extent());
  ASSERT_EQ(s.rows(), x.extent());
  ASSERT_EQ(s.columns(), y.extent());
}

template <size_t rank>
void test_all_unknown_strides() {
  std::array<dim<>, rank> a;
  std::array<dim<>, rank> b;
  index_t stride = 1;
  for (size_t d = 0; d < rank; d++) {
    a[d] = dim<>(d);
    b[d] = a[d];
    b[d].set_stride(stride);
    stride *= std::max(static_cast<index_t>(1), a[d].extent());
  }
  shape_of_rank<rank> s_all_unknown(internal::array_to_tuple(a));
  shape_of_rank<rank> s_all_unknown_resolved(internal::array_to_tuple(b));
  s_all_unknown.resolve();
  s_all_unknown_resolved.resolve();
  ASSERT_EQ(s_all_unknown, s_all_unknown_resolved);
}

size_t factorial(size_t x) {
  if (x <= 1) { return 1; }
  return x * factorial(x - 1);
}

template <size_t rank>
void test_one_dense_stride() {
  for (size_t known = 0; known < rank; known++) {
    std::array<dim<>, rank> a;
    for (size_t d = 0; d < rank; d++) {
      a[d] = dim<>(d + 1);
      if (d == known) {
        // This is the dimension we know.
        a[d].set_stride(1);
      }
    }
    shape_of_rank<rank> s_one_dense(internal::array_to_tuple(a));
    s_one_dense.resolve();
    ASSERT_EQ(s_one_dense.size(), factorial(rank));
    ASSERT_EQ(s_one_dense.dim(known).stride(), 1);
    ASSERT(s_one_dense.is_compact());
    ASSERT(s_one_dense.is_one_to_one());
  }
}

template <size_t rank>
void test_auto_strides() {
  test_all_unknown_strides<rank>();
  test_one_dense_stride<rank>();
}

template <size_t Rank>
void check_resolved_strides(shape_of_rank<Rank> shape, const std::vector<index_t>& strides) {
  shape.resolve();
  for (size_t i = 0; i < strides.size(); i++) {
    ASSERT_EQ(shape.dim(i).stride(), strides[i]);
  }
}

TEST(auto_strides) {
  check_resolved_strides<1>({{3, 5, dynamic}}, {1});
  check_resolved_strides<2>({5, 10}, {1, 5});

  // Small interleaved.
  // TODO: This test would be nice to enable, but the automatic strides are too clever.
  // x is given a stride of 1, which is safe and correct, but annoying.
  // check_resolved_strides<3>({1, 1, {0, 2, 1}}, {2, 2, 1});

  // Interleaved with row stride.
  check_resolved_strides<3>({5, {0, 4, 20}, {0, 3, 1}}, {3, 20, 1});

  // Interleaved with row stride dense.
  check_resolved_strides<3>({5, {0, 4, 15}, {0, 3, 1}}, {3, 15, 1});

  // Interleaved with row stride oops.
  check_resolved_strides<3>({5, {0, 4, 14}, {0, 3, 1}}, {56, 14, 1});

  test_auto_strides<1>();
  test_auto_strides<2>();
  test_auto_strides<3>();
  test_auto_strides<4>();
  test_auto_strides<5>();
  test_auto_strides<6>();
  test_auto_strides<7>();
  test_auto_strides<8>();
  test_auto_strides<9>();
  test_auto_strides<10>();
}

TEST(broadcast_dim) {
  dim<> x(0, 10, 1);
  broadcast_dim<> y;
  shape<dim<>, broadcast_dim<>> s(x, y);
  for (int i = 0; i < 10; i++) {
    for (int j : x) {
      ASSERT_EQ(s(j, i), j);
    }
  }
}

TEST(clamp) {
  dim<> x(5, 10, 1);
  for (int i = -10; i < 20; i++) {
    int correct = std::max(std::min(i, 14), 5);
    ASSERT_EQ(clamp(i, x), correct);
  }
}

TEST(for_all_indices_scalar) {
  shape<> s;
  int count = 0;
  move_only token;
  for_all_indices(s, [&, token = std::move(token)]() {
    count++;
    assert_used(token);
  });
  ASSERT_EQ(count, 1);
}

TEST(for_all_indices_1d) {
  dense_shape<1> s(20);
  int expected_flat_offset = 0;
  for_all_indices(s, [&](int x) {
    ASSERT_EQ(s(x), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_all_indices loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 20);
}

TEST(for_all_indices_2d) {
  dense_shape<2> s(10, 4);
  s.resolve();
  int expected_flat_offset = 0;
  for_all_indices(s, [&](int x, int y) {
    ASSERT_EQ(s(x, y), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_all_indices loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 40);
}

TEST(for_all_indices_3d) {
  dense_shape<3> s(3, 5, 8);
  s.resolve();
  int expected_flat_offset = 0;
  move_only token;
  for_all_indices(s, [&, token = std::move(token)](int x, int y, int z) {
    ASSERT_EQ(s(x, y, z), expected_flat_offset);
    expected_flat_offset++;
    assert_used(token);
  });
  // Ensure the for_all_indices loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 120);
}

TEST(for_all_indices_3d_reordered) {
  shape_of_rank<3> s(3, 5, {0, 8, 1});
  s.resolve();
  int expected_flat_offset = 0;
  move_only token;
  for_all_indices<2, 0, 1>(s, [&, token = std::move(token)](int x, int y, int z) {
    ASSERT_EQ(s(x, y, z), expected_flat_offset);
    expected_flat_offset++;
    assert_used(token);
  });
  // Ensure the for_all_indices loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 120);
}

TEST(for_all_indices_2d_of_3d) {
  shape_of_rank<3> s(3, 5, {0, 8, 1});
  s.resolve();
  int expected_flat_offset = 0;
  for_all_indices<0, 1>(s, [&](int x, int y) {
    ASSERT_EQ(s(x, y, 0), expected_flat_offset);
    expected_flat_offset += s.x().stride();
  });
  // Ensure the for_all_indices loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 120);
}

TEST(for_each_index_scalar) {
  shape<> s;
  int count = 0;
  move_only token;
  for_each_index(s, [&, token = std::move(token)](std::tuple<>) {
    count++;
    assert_used(token);
  });
  ASSERT_EQ(count, 1);
}

TEST(for_each_index_1d) {
  dense_shape<1> s(20);
  int expected_flat_offset = 0;
  for_each_index(s, [&](std::tuple<int> i) {
    ASSERT_EQ(s(i), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 20);
}

TEST(for_each_index_2d) {
  dense_shape<2> s(10, 4);
  s.resolve();
  int expected_flat_offset = 0;
  for_each_index(s, [&](std::tuple<int, int> i) {
    ASSERT_EQ(s(i), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 40);
}

TEST(for_each_index_3d) {
  dense_shape<3> s(3, 5, 8);
  s.resolve();
  int expected_flat_offset = 0;
  move_only token;
  for_each_index(s, [&, token = std::move(token)](std::tuple<int, int, int> i) {
    ASSERT_EQ(s(i), expected_flat_offset);
    expected_flat_offset++;
    assert_used(token);
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 120);
}

TEST(for_each_index_3d_reorderd) {
  shape_of_rank<3> s(3, 5, {0, 8, 1});
  s.resolve();
  int expected_flat_offset = 0;
  move_only token;
  for_each_index<2, 0, 1>(s, [&, token = std::move(token)](std::tuple<int, int, int> i) {
    ASSERT_EQ(s(i), expected_flat_offset);
    expected_flat_offset++;
    assert_used(token);
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 120);
}

TEST(for_each_index_2d_of_3d) {
  shape_of_rank<3> s(3, 5, {0, 8, 1});
  s.resolve();
  int expected_flat_offset = 0;
  for_each_index<0, 1>(s, [&](std::tuple<int, int> i) {
    ASSERT_EQ(s(std::get<0>(i), std::get<1>(i), 0), expected_flat_offset);
    expected_flat_offset += s.x().stride();
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 120);
}

TEST(dim_is_in_range) {
  dim<> x(2, 5);

  for (int i = 2; i < 7; i++) {
    ASSERT(x.is_in_range(i));
  }
  ASSERT(!x.is_in_range(1));
  ASSERT(!x.is_in_range(8));

  ASSERT(x.is_in_range(x));
  ASSERT(!x.is_in_range(interval<>(1, 2)));
  ASSERT(!x.is_in_range(interval<>(8, 2)));
}

TEST(shape_is_in_range_1d) {
  dim<> x(2, 5);
  shape<dim<>> s(x);

  for (int i = 2; i < 7; i++) {
    ASSERT(s.is_in_range(i));
  }
  ASSERT(!s.is_in_range(1));
  ASSERT(!s.is_in_range(8));

  ASSERT(s.is_in_range(x));
  ASSERT(!s.is_in_range(interval<>(0, 2)));
  ASSERT(!s.is_in_range(interval<>(8, 12)));
}

TEST(shape_is_in_range_2d) {
  dim<> x(2, 5);
  dim<> y(-3, 6);
  shape<dim<>, dim<>> s(x, y);

  for (int i = -3; i < 3; i++) {
    for (int j = 2; j < 7; j++) {
      ASSERT(s.is_in_range(j, i));
    }
  }
  ASSERT(!s.is_in_range(1, 0));
  ASSERT(!s.is_in_range(2, -4));

  ASSERT(!s.is_in_range(8, 0));
  ASSERT(!s.is_in_range(2, 4));

  ASSERT(s.is_in_range(x, y));
  ASSERT(!s.is_in_range(1, y));
  ASSERT(!s.is_in_range(x, -4));
}

TEST(shape_conversion) {
  dense_dim<> x_dense(0, 10);
  dim<> x = x_dense;

  assert_dim_eq(x, dim<>(0, 10, 1));

  dense_shape<2> static_dense({0, 10}, {1, 5});
  shape_of_rank<2> dense = static_dense;
  ASSERT_EQ(dense, static_dense);

  static_dense = dense;
  ASSERT_EQ(dense, static_dense);

  dense_shape<2> static_dense2(dense);
  ASSERT_EQ(dense, static_dense2);

  ASSERT(is_compatible<dense_shape<2>>(dense));

  shape_of_rank<2> sparse({0, 10, 2}, {1, 5, 20});
  ASSERT(!is_compatible<dense_shape<2>>(sparse));

  dense_shape<3> uprank = convert_shape<dense_shape<3>>(dense);
  ASSERT_EQ(uprank.z().min(), 0);
  ASSERT_EQ(uprank.z().extent(), 1);
}

TEST(shape_transpose) {
  dense_shape<3> s(3, 5, 8);
  shape<dim<>, dim<>, dense_dim<>> transposed = transpose<1, 2, 0>(s);
  ASSERT_EQ(transposed.template dim<0>().extent(), 5);
  ASSERT_EQ(transposed.template dim<1>().extent(), 8);
  ASSERT_EQ(transposed.template dim<2>().extent(), 3);

  dense_shape<2> reordered = reorder<2, 0>(transposed);
  ASSERT_EQ(reordered.template dim<0>().extent(), 3);
  ASSERT_EQ(reordered.template dim<1>().extent(), 5);
}

TEST(shape_optimize) {
  dim<> dummy_dim(0, 1, 0);

  shape_of_rank<3> a({0, 5, 21}, {0, 7, 3}, {5, 3, 1});
  shape_of_rank<3> a_optimized({5, 105, 1}, dummy_dim, dummy_dim);
  assert_shapes_eq(internal::dynamic_optimize_shape(a), a_optimized);

  shape_of_rank<3> b({0, 5, 42}, {3, 7, 6}, {0, 3, 2});
  shape_of_rank<3> b_optimized({9, 105, 2}, dummy_dim, dummy_dim);
  assert_shapes_eq(internal::dynamic_optimize_shape(b), b_optimized);

  shape_of_rank<3> c({0, 5, 40}, {0, 7, 3}, {0, 2, 1});
  shape_of_rank<3> c_optimized({0, 2, 1}, {0, 7, 3}, {0, 5, 40});
  assert_shapes_eq(internal::dynamic_optimize_shape(c), c_optimized);

  shape_of_rank<3> d({0, 5, 28}, {0, 7, 4}, {0, 3, 1});
  shape_of_rank<3> d_optimized({0, 3, 1}, {0, 35, 4}, dummy_dim);
  assert_shapes_eq(internal::dynamic_optimize_shape(d), d_optimized);

  shape_of_rank<10> e(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  e.resolve();
  shape_of_rank<10> e2 = reorder<9, 5, 3, 7, 2, 8, 4, 6, 0, 1>(e);
  shape_of_rank<10> e_optimized({0, 3628800, 1}, dummy_dim, dummy_dim, dummy_dim, dummy_dim,
      dummy_dim, dummy_dim, dummy_dim, dummy_dim, dummy_dim);
  assert_shapes_eq(internal::dynamic_optimize_shape(e), e_optimized);
  assert_shapes_eq(internal::dynamic_optimize_shape(e2), e_optimized);

  shape_of_rank<2> f({0, 2}, {1, 2});
  shape_of_rank<2> f_optimized({2, 4, 1}, dummy_dim);
  f.resolve();
  assert_shapes_eq(internal::dynamic_optimize_shape(f), f_optimized);

  shape_of_rank<2> g({1, 2}, {1, 2});
  shape_of_rank<2> g_optimized({3, 4, 1}, dummy_dim);
  g.resolve();
  assert_shapes_eq(internal::dynamic_optimize_shape(g), g_optimized);
}

TEST(shape_make_compact) {
  shape<dim<>> s1({3, 5, 2});
  shape<dense_dim<>> s1_compact({3, 5, 1});
  assert_shapes_eq(make_compact(s1), s1_compact);

  shape<dim<>, dim<>> s2({3, 5, 8}, {1, 4, 1});
  shape<dense_dim<>, dim<>> s2_compact({3, 5, 1}, {1, 4, 5});
  assert_shapes_eq(make_compact(s2), s2_compact);

  shape<dim<>, dense_dim<>> s3({3, 5, 8}, {1, 4});
  shape<dim<>, dense_dim<>> s3_compact({3, 5, 4}, {1, 4});
  assert_shapes_eq(make_compact(s3), s3_compact);

  shape<dim<>, dense_dim<dynamic, 4>> s4({5, 3, 8}, {2, 4});
  shape<strided_dim<4>, dense_dim<dynamic, 4>> s4_compact({5, 3}, {2, 4});
  assert_shapes_eq(make_compact(s4), s4_compact);

  // We can't make the first dimenion static stride 1, because
  // we don't know its extent when assigning the static strides.
  shape<dim<>, strided_dim<8>> s5({5, 8}, {2, 4});
  shape<dim<>, strided_dim<8>> s5_compact({5, 8, 1}, {2, 4});
  assert_shapes_eq(make_compact(s5), s5_compact);

  // TODO: Try to give this first dimension static stride 1.
  shape<dim<0, 8>, dim<0, 4, 8>> s6({}, {});
  shape<dim<0, 8>, dim<0, 4, 8>> s6_compact({0, 8, 1}, {});
  assert_shapes_eq(make_compact(s6), s6_compact);
}

template <typename Shape>
void test_number_theory(Shape s) {
  s.resolve();

  std::vector<int> addresses(static_cast<size_t>(s.flat_extent()), 0);
  for_each_index(s, [&](const typename Shape::index_type& i) {
    addresses[static_cast<size_t>(s(i) - s.flat_min())] += 1;
  });
  bool is_compact = std::all_of(addresses.begin(), addresses.end(), [](int c) { return c >= 1; });
  bool is_one_to_one =
      std::all_of(addresses.begin(), addresses.end(), [](int c) { return c <= 1; });

  ASSERT_EQ(s.is_compact(), is_compact);
  ASSERT_EQ(s.is_one_to_one(), is_one_to_one);
}

TEST(shape_number_theory) {
  test_number_theory(shape_of_rank<2>({1, 10}, {3, 5}));
  test_number_theory(shape_of_rank<2>({-1, 10}, {3, 5, -1}));
  test_number_theory(shape_of_rank<2>({-2, 10, 6}, {3, 5}));
  test_number_theory(shape_of_rank<3>({0, 4, 4}, {0, 4, 2}, {0, 4, 1}));
  // TODO: https://github.com/dsharlet/array/issues/2
  // test_number_theory(shape_of_rank<2>({0, 4, 4}, {0, 4, 4}));
}

} // namespace nda

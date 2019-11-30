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
    shape<dim<>> s = make_shape(x);
    for (int i : x) {
      ASSERT_EQ(s(i), i * stride);
    }
  }
}

TEST(shape_1d_dense) {
  dense_dim<> x(0, 10);
  shape<dense_dim<>> s = make_shape(x);
  for (int i : x) {
    ASSERT_EQ(s(i), i);
  }
}

TEST(shape_2d) {
  dense_dim<> x(0, 10);
  dim<> y(0, 5, x.extent());
  shape<dense_dim<>, dim<>> s = make_shape(x, y);
  for (int i : y) {
    for (int j : x) {
      ASSERT_EQ(s(j, i), i * x.extent() + j);
    }
  }
}

TEST(shape_2d_negative_stride) {
  dense_dim<> x(0, 10);
  dim<> y(0, 5, -x.extent());
  shape<dense_dim<>, dim<>> s = make_shape(x, y);
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
  ASSERT_EQ(s2.x().stride(), 3);
  ASSERT_EQ(s2.y().stride(), 30);
}

TEST(make_dense_shape_1d) {
  dense_shape<1> s = make_dense_shape(10);
  dense_dim<> x = s.template dim<0>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);
}

TEST(make_dense_shape_2d) {
  dense_shape<2> s(10, 5);
  dense_dim<> x = s.template dim<0>();
  dim<> y = s.template dim<1>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);
  ASSERT_EQ(y.min(), 0);
  ASSERT_EQ(y.extent(), 5);
  ASSERT_EQ(y.stride(), 10);

  ASSERT_EQ(s.width(), x.extent());
  ASSERT_EQ(s.height(), y.extent());
  ASSERT_EQ(s.rows(), x.extent());
  ASSERT_EQ(s.columns(), y.extent());
}

TEST(make_dense_shape_3d) {
  dense_shape<3> s(10, 5, 20);
  dense_dim<> x = s.template dim<0>();
  dim<> y = s.template dim<1>();
  dim<> z = s.template dim<2>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);
  ASSERT_EQ(y.min(), 0);
  ASSERT_EQ(y.extent(), 5);
  ASSERT_EQ(y.stride(), 10);
  ASSERT_EQ(z.min(), 0);
  ASSERT_EQ(z.extent(), 20);
  ASSERT_EQ(z.stride(), 50);

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
  for (int d = 0; d < rank; d++) {
    a[d] = dim<>(d);
    b[d] = a[d];
    b[d].set_stride(stride);
    stride *= std::max(static_cast<index_t>(1), a[d].extent());
  }
  shape_of_rank<rank> s_all_unknown(internal::array_to_tuple(a));
  shape_of_rank<rank> s_all_unknown_resolved(internal::array_to_tuple(b));
  ASSERT_EQ(s_all_unknown, s_all_unknown_resolved);
}

index_t factorial(index_t x) {
  if (x <= 1) {
    return 1;
  }
  return x * factorial(x - 1);
}

template <size_t rank>
void test_one_dense_stride() {
  for (int known = 0; known < rank; known++) {
    std::array<dim<>, rank> a;
    for (int d = 0; d < rank; d++) {
      a[d] = dim<>(d + 1);
      if (d == known) {
        // This is the dimension we know.
        a[d].set_stride(1);
      }
    }
    shape_of_rank<rank> s_one_dense(internal::array_to_tuple(a));
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

TEST(auto_strides) {
  shape<dim<>> s1(dim<>(3, 5, UNK));
  shape<dim<>> s1_resolved(dim<>(3, 5, 1));
  ASSERT_EQ(s1, s1_resolved);

  shape<dim<>, dim<>> s2(5, 10);
  shape<dim<>, dim<>> s2_resolved({0, 5, 1}, {0, 10, 5});
  ASSERT_EQ(s2, s2_resolved);

  // TODO: This test would be nice to enable, but the automatic strides are too clever.
  // x is given a stride of 1, which is safe and correct, but annoying.
  //shape<dim<>, dim<>, dim<>> s_small_interleaved(1, 1, {0, 2, 1});
  //shape<dim<>, dim<>, dim<>> s_small_interleaved_resolved({0, 1, 2}, {0, 1, 2}, {0, 2, 1});
  //ASSERT_EQ(s_small_interleaved, s_small_interleaved_resolved);

  shape<dim<>, dim<>, dim<>> s_interleaved_with_row_stride(5, {0, 4, 20}, {0, 3, 1});
  shape<dim<>, dim<>, dim<>> s_interleaved_with_row_stride_resolved({0, 5, 3}, {0, 4, 20}, {0, 3, 1});
  ASSERT_EQ(s_interleaved_with_row_stride, s_interleaved_with_row_stride_resolved);

  shape<dim<>, dim<>, dim<>> s_interleaved_with_row_stride_dense(5, {0, 4, 15}, {0, 3, 1});
  shape<dim<>, dim<>, dim<>> s_interleaved_with_row_stride_dense_resolved({0, 5, 3}, {0, 4, 15}, {0, 3, 1});
  ASSERT_EQ(s_interleaved_with_row_stride_dense, s_interleaved_with_row_stride_dense_resolved);

  shape<dim<>, dim<>, dim<>> s_interleaved_with_row_stride_oops(5, {0, 4, 14}, {0, 3, 1});
  shape<dim<>, dim<>, dim<>> s_interleaved_with_row_stride_oops_resolved({0, 5, 56}, {0, 4, 14}, {0, 3, 1});
  ASSERT_EQ(s_interleaved_with_row_stride_oops, s_interleaved_with_row_stride_oops_resolved);

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
  shape<dim<>, broadcast_dim<>> s = make_shape(x, y);
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
  for_all_indices(s, [&]() {
    count++;
  });
  ASSERT_EQ(count, 1);
}

TEST(for_all_indices_1d) {
  dense_shape<1> s = make_dense_shape(20);
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
  int expected_flat_offset = 0;
  for_all_indices(s, [&](int x, int y, int z) {
    ASSERT_EQ(s(x, y, z), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_all_indices loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 120);
}

TEST(for_each_index_scalar) {
  shape<> s;
  int count = 0;
  for_each_index(s, [&](std::tuple<>) {
    count++;
  });
  ASSERT_EQ(count, 1);
}

TEST(for_each_index_1d) {
  dense_shape<1> s = make_dense_shape(20);
  int expected_flat_offset = 0;
  for_each_index(s, [&](std::tuple<int> x) {
    ASSERT_EQ(s(x), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 20);
}

TEST(for_each_index_2d) {
  dense_shape<2> s(10, 4);
  int expected_flat_offset = 0;
  for_each_index(s, [&](std::tuple<int, int> x) {
    ASSERT_EQ(s(x), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 40);
}

TEST(for_each_index_3d) {
  dense_shape<3> s(3, 5, 8);
  int expected_flat_offset = 0;
  for_each_index(s, [&](std::tuple<int, int, int> x) {
    ASSERT_EQ(s(x), expected_flat_offset);
    expected_flat_offset++;
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
}

TEST(shape_is_in_range_1d) {
  dim<> x(2, 5);
  shape<dim<>> s = make_shape(x);

  for (int i = 2; i < 7; i++) {
    ASSERT(s.is_in_range(i));
  }
  ASSERT(!s.is_in_range(1));
  ASSERT(!s.is_in_range(8));
}

TEST(shape_is_in_range_2d) {
  dim<> x(2, 5);
  dim<> y(-3, 6);
  shape<dim<>, dim<>> s = make_shape(x, y);

  for (int i = -3; i < 3; i++) {
    for (int j = 2; j < 7; j++) {
      ASSERT(s.is_in_range(j, i));
    }
  }
  ASSERT(!s.is_in_range(1, 0));
  ASSERT(!s.is_in_range(2, -4));

  ASSERT(!s.is_in_range(8, 0));
  ASSERT(!s.is_in_range(2, 4));
}

TEST(shape_conversion) {
  dense_dim<> x_dense(0, 10);
  dim<> x = x_dense;

  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);

  dense_shape<2> static_dense(dense_dim<>(0, 10), dim<>(1, 5));
  shape_of_rank<2> dense = static_dense;
  ASSERT_EQ(dense, static_dense);

  static_dense = dense;
  ASSERT_EQ(dense, static_dense);

  dense_shape<2> static_dense2(dense);
  ASSERT_EQ(dense, static_dense2);

  ASSERT(is_convertible<dense_shape<2>>(dense));

  shape_of_rank<2> sparse(dim<>(0, 10, 2), dim<>(1, 5, 20));
  ASSERT(!is_convertible<dense_shape<2>>(sparse));
}

TEST(shape_transpose) {
  dense_shape<3> s(3, 5, 8);
  shape<dim<>, dim<>, dense_dim<>> transposed = reorder<1, 2, 0>(s);
  ASSERT_EQ(transposed.template dim<0>().extent(), 5);
  ASSERT_EQ(transposed.template dim<1>().extent(), 8);
  ASSERT_EQ(transposed.template dim<2>().extent(), 3);

  shape<dim<>, dim<>, dense_dim<>> interleaved(3, 5, 4);
  ASSERT(interleaved.is_compact());
  int expected_flat_offset = 0;
  for_all_indices(reorder<2, 0, 1>(interleaved), [&](int c, int x, int y) {
    ASSERT_EQ(interleaved(x, y, c), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 60);
}

TEST(shape_optimize) {
  shape_of_rank<3> a({0, 5, 21}, {0, 7, 3}, {5, 3, 1});
  shape_of_rank<3> a_optimized({5, 105, 1}, {0, 1, 105}, {0, 1, 105});
  ASSERT_EQ(internal::dynamic_optimize_shape(a), a_optimized);

  shape_of_rank<3> b({0, 5, 42}, {3, 7, 6}, {0, 3, 2});
  shape_of_rank<3> b_optimized({9, 105, 2}, {0, 1, 210}, {0, 1, 210});
  ASSERT_EQ(internal::dynamic_optimize_shape(b), b_optimized);

  shape_of_rank<3> c({0, 5, 40}, {0, 7, 3}, {0, 2, 1});
  shape_of_rank<3> c_optimized({0, 2, 1}, {0, 7, 3}, {0, 5, 40});
  ASSERT_EQ(internal::dynamic_optimize_shape(c), c_optimized);

  shape_of_rank<3> d({0, 5, 28}, {0, 7, 4}, {0, 3, 1});
  shape_of_rank<3> d_optimized({0, 3, 1}, {0, 35, 4}, {0, 1, 140});
  ASSERT_EQ(internal::dynamic_optimize_shape(d), d_optimized);

  shape_of_rank<10> e(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  shape_of_rank<10> e2 = reorder<9, 5, 3, 7, 2, 8, 4, 6, 0, 1>(e);
  dim<> e_optimized_dim(0, 1, 3628800);
  shape_of_rank<10> e_optimized(
    3628800,
    e_optimized_dim, e_optimized_dim, e_optimized_dim,
    e_optimized_dim, e_optimized_dim, e_optimized_dim,
    e_optimized_dim, e_optimized_dim, e_optimized_dim);
  ASSERT_EQ(internal::dynamic_optimize_shape(e), e_optimized);
  ASSERT_EQ(internal::dynamic_optimize_shape(e2), e_optimized);

  shape_of_rank<2> f({0, 2}, {1, 2});
  shape_of_rank<2> f_optimized({2, 4, 1}, {0, 1, 4});
  ASSERT_EQ(internal::dynamic_optimize_shape(f), f_optimized);

  shape_of_rank<2> g({1, 2}, {1, 2});
  shape_of_rank<2> g_optimized({3, 4, 1}, {0, 1, 4});
  ASSERT_EQ(internal::dynamic_optimize_shape(g), g_optimized);
}

TEST(shape_make_compact) {
  shape<dim<>> s1(dim<>(3, 5, 2));
  shape<dim<>> s1_compact(dim<>(3, 5, 1));
  ASSERT_EQ(make_compact(s1), s1_compact);

  shape<dim<>, dim<>> s2(dim<>(3, 5, 8), dim<>(1, 4, 1));
  shape<dim<>, dim<>> s2_compact(dim<>(3, 5, 1), dim<>(1, 4, 5));
  ASSERT_EQ(make_compact(s2), s2_compact);

  shape<dim<>, dense_dim<>> s3(dim<>(3, 5, 8), dense_dim<>(1, 4));
  shape<dim<>, dense_dim<>> s3_compact(dim<>(3, 5, 4), dense_dim<>(1, 4));
  ASSERT_EQ(make_compact(s3), s3_compact);
}

TEST(shape_intersect) {
  shape<> s0;
  shape<dim<>> s1({1, 9});
  shape<dim<0, UNK>, dim<>> s2({0, 12}, {-10, 100});
  shape<dense_dim<-1, 5>, dim<>, dim<>> s3({-1, 5}, {10, 20}, {-100, 1000});
  shape<dense_dim<0, 3>, dim<>, dim<>, dim<>> s4({0, 3}, {20, 40}, {-100, 1000}, {0, 1});

  shape<> s0_s1;
  shape<dim<>> s1_s2({1, 9});
  shape<dim<0, UNK>, dim<>> s2_s3({0, 4}, {10, 20});
  shape<dim<0, 3>, dim<>, dim<>> s3_s4({0, 3}, {20, 10}, {-100, 1000});
  ASSERT_EQ(intersect(s0, s1), s0_s1);
  ASSERT_EQ(intersect(s1, s2), s1_s2);
  ASSERT_EQ(intersect(s2, s3), s2_s3);
  ASSERT_EQ(intersect(s3, s4), s3_s4);
}

template <typename Shape>
void test_number_theory(const Shape& s) {
  std::vector<int> addresses(static_cast<size_t>(s.flat_extent()), 0);
  for_each_index(s, [&](const typename Shape::index_type& i) {
    addresses[static_cast<size_t>(s(i) - s.flat_min())] += 1;
  });
  bool is_compact = std::all_of(addresses.begin(), addresses.end(), [](int c) { return c >= 1; });
  bool is_one_to_one = std::all_of(addresses.begin(), addresses.end(), [](int c) { return c <= 1; });

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

}  // namespace nda

#include "array.h"
#include "test.h"

namespace array {

TEST(euclidean_div_mod) {
  const int values[] = {
    -1000, -100, -10, -2, -1, 0, 1, 2, 10, 100, 1000,
  };

  for (int b : values) {
    if (b == 0) continue;
    for (int a : values) {
      int q = internal::euclidean_div(a, b);
      int r = internal::euclidean_mod(a, b);
      ASSERT_EQ(q * b + r, a);
      ASSERT(0 <= r && r < std::abs(b));
    }
  }
}

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

TEST(auto_strides) {
  shape_of_rank<3> s(10, 20, 3);
  dim<> x = s.template dim<0>();
  dim<> y = s.template dim<1>();
  dim<> z = s.template dim<2>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);
  ASSERT_EQ(y.min(), 0);
  ASSERT_EQ(y.extent(), 20);
  ASSERT_EQ(y.stride(), 10);
  ASSERT_EQ(z.min(), 0);
  ASSERT_EQ(z.extent(), 3);
  ASSERT_EQ(z.stride(), 200);
}

TEST(auto_strides_interleaved) {
  shape<dim<>, dim<>, dense_dim<>> s(10, 20, 3);
  dim<> x = s.template dim<0>();
  dim<> y = s.template dim<1>();
  dense_dim<> z = s.template dim<2>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 3);
  ASSERT_EQ(y.min(), 0);
  ASSERT_EQ(y.extent(), 20);
  ASSERT_EQ(y.stride(), 30);
  ASSERT_EQ(z.min(), 0);
  ASSERT_EQ(z.extent(), 3);
  ASSERT_EQ(z.stride(), 1);
}

TEST(folded_dim) {
  dim<> x(0, 10, 1);
  folded_dim<> y(4, 10);
  shape<dim<>, folded_dim<>> s = make_shape(x, y);
  for (int i = 0; i < 10; i++) {
    for (int j : x) {
      ASSERT_EQ(s(j, i), (i % 4) * 10 + j);
    }
  }
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
    ASSERT(clamp(i, x) == correct);
  }
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
  ASSERT(dense == static_dense);

  static_dense = dense;
  ASSERT(dense == static_dense);

  dense_shape<2> static_dense2(dense);
  ASSERT(dense == static_dense2);

  ASSERT(is_shape_compatible<dense_shape<2>>(dense));

  shape_of_rank<2> sparse(dim<>(0, 10, 2), dim<>(1, 5, 20));
  ASSERT(!is_shape_compatible<dense_shape<2>>(sparse));
}

TEST(shape_transpose) {
  dense_shape<3> s(3, 5, 8);
  shape<dim<>, dim<>, dense_dim<>> transposed = permute<1, 2, 0>(s);
  ASSERT_EQ(transposed.template dim<0>().extent(), 5);
  ASSERT_EQ(transposed.template dim<1>().extent(), 8);
  ASSERT_EQ(transposed.template dim<2>().extent(), 3);

  shape<dim<>, dim<>, dense_dim<>> interleaved(3, 5, 4);
  ASSERT(interleaved.is_compact());
  int expected_flat_offset = 0;
  for_all_indices(permute<2, 0, 1>(interleaved), [&](int c, int x, int y) {
    ASSERT_EQ(interleaved(x, y, c), expected_flat_offset);
    expected_flat_offset++;
  });
  // Ensure the for_each_index loop above actually ran.
  ASSERT_EQ(expected_flat_offset, 60);
}

TEST(shape_optimize) {
  shape_of_rank<3> a({0, 5, 21}, {0, 7, 3}, {5, 3, 1});
  shape_of_rank<3> a_optimized({5, 105, 1}, {0, 1, 105}, {0, 1, 105});
  ASSERT(internal::optimize_shape(a) == a_optimized);

  shape_of_rank<3> b({0, 5, 42}, {3, 7, 6}, {0, 3, 2});
  shape_of_rank<3> b_optimized({18, 105, 2}, {0, 1, 210}, {0, 1, 210});
  ASSERT(internal::optimize_shape(b) == b_optimized);

  shape_of_rank<3> c({0, 5, 40}, {0, 7, 3}, {0, 2, 1});
  shape_of_rank<3> c_optimized({0, 2, 1}, {0, 7, 3}, {0, 5, 40});
  ASSERT(internal::optimize_shape(c) == c_optimized);

  shape_of_rank<3> d({0, 5, 28}, {0, 7, 4}, {0, 3, 1});
  shape_of_rank<3> d_optimized({0, 3, 1}, {0, 35, 4}, {0, 1, 140});
  ASSERT(internal::optimize_shape(d) == d_optimized);

  shape_of_rank<10> e(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
  shape_of_rank<10> e2 = permute<9, 5, 3, 7, 2, 8, 4, 6, 0, 1>(e);
  shape_of_rank<10> e_optimized(3628800, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  ASSERT(internal::optimize_shape(e) == e_optimized);
  ASSERT(internal::optimize_shape(e2) == e_optimized);

  shape_of_rank<2> f({0, 2}, {1, 2});
  shape_of_rank<2> f_optimized({2, 4}, {0, 1});
  ASSERT(internal::optimize_shape(f) == f_optimized);

  shape_of_rank<2> g({1, 2}, {1, 2});
  shape_of_rank<2> g_optimized({3, 4}, {0, 1});
  ASSERT(internal::optimize_shape(g) == g_optimized);

}

}  // namespace array

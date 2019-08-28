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
    auto s = make_shape(x);
    for (int i : x) {
      ASSERT_EQ(s(i), i * stride);
    }
  }
}

TEST(shape_1d_dense) {
  dense_dim<> x(0, 10);
  auto s = make_shape(x);
  for (int i : x) {
    ASSERT_EQ(s(i), i);
  }
}

TEST(shape_2d) {
  dense_dim<> x(0, 10);
  dim<> y(0, 5, x.extent());
  auto s = make_shape(x, y);
  for (int i : y) {
    for (int j : x) {
      ASSERT_EQ(s(j, i), i * x.extent() + j);
    }
  }
}

TEST(make_dense_shape_1d) {
  auto s = make_dense_shape(10);
  auto x = s.template dim<0>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);
}

TEST(make_dense_shape_2d) {
  dense_shape<2> s(10, 5);
  auto x = s.template dim<0>();
  auto y = s.template dim<1>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);
  ASSERT_EQ(y.min(), 0);
  ASSERT_EQ(y.extent(), 5);
  ASSERT_EQ(y.stride(), 10);
}

TEST(make_dense_shape_3d) {
  dense_shape<3> s(10, 5, 20);
  auto x = s.template dim<0>();
  auto y = s.template dim<1>();
  auto z = s.template dim<2>();
  ASSERT_EQ(x.min(), 0);
  ASSERT_EQ(x.extent(), 10);
  ASSERT_EQ(x.stride(), 1);
  ASSERT_EQ(y.min(), 0);
  ASSERT_EQ(y.extent(), 5);
  ASSERT_EQ(y.stride(), 10);
  ASSERT_EQ(z.min(), 0);
  ASSERT_EQ(z.extent(), 20);
  ASSERT_EQ(z.stride(), 50);
}

TEST(auto_strides) {
  shape_of_rank<3> s(10, 20, 3);
  auto x = s.template dim<0>();
  auto y = s.template dim<1>();
  auto z = s.template dim<2>();
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
  auto x = s.template dim<0>();
  auto y = s.template dim<1>();
  auto z = s.template dim<2>();
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
  auto s = make_shape(x, y);
  for (int i = 0; i < 10; i++) {
    for (int j : x) {
      ASSERT_EQ(s(j, i), (i % 4) * 10 + j);
    }
  }
}

TEST(broadcast_dim) {
  dim<> x(0, 10, 1);
  broadcast_dim<> y;
  auto s = make_shape(x, y);
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
  auto s = make_dense_shape(20);
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
  auto s = make_dense_shape(20);
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
  auto s = make_shape(x);

  for (int i = 2; i < 7; i++) {
    ASSERT(s.is_in_range(i));
  }
  ASSERT(!s.is_in_range(1));
  ASSERT(!s.is_in_range(8));
}

TEST(shape_is_in_range_2d) {
  dim<> x(2, 5);
  dim<> y(-3, 6);
  auto s = make_shape(x, y);

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

  // TODO: Enabling this leads to lots of overload ambiguity.
  //dense_shape<2> static_dense(dense_dim<>(0, 10), dim<>(1, 5));
  //shape_of_rank<2> dense = static_dense;
}

}  // namespace array

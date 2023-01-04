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

#include "array/array.h"
#include "test.h"

namespace nda {

TEST(array_default_constructor) {
  // GCC 9.3 struggles to disambiguate this, clang is fine.
  dense_array<int, 1> a(dense_shape<1>{10});
  for (int x = 0; x < 10; x++) {
    ASSERT_EQ(a(x), 0);
  }

  dense_array<int, 2> b({7, 3});
  ASSERT_EQ(b.width(), 7);
  ASSERT_EQ(b.height(), 3);
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 7; x++) {
      ASSERT_EQ(b(x, y), 0);
    }
  }

  dense_array<int, 3> c({5, 9, 3});
  ASSERT_EQ(c.width(), 5);
  ASSERT_EQ(c.height(), 9);
  ASSERT_EQ(c.channels(), 3);
  for (int z = 0; z < 3; z++) {
    for (int y = 0; y < 9; y++) {
      for (int x = 0; x < 5; x++) {
        ASSERT_EQ(c(x, y, z), 0);
      }
    }
  }

  array<int, shape<dim<>, dim<>>> sparse({{-2, 5, 2}, {4, 10, 20}});
  ASSERT_EQ(sparse.rows(), 5);
  ASSERT_EQ(sparse.columns(), 10);
  for (int y = 4; y < 14; y++) {
    for (int x = -2; x < 3; x++) {
      ASSERT_EQ(sparse(x, y), 0);
    }
  }

  sparse.clear();
  ASSERT(sparse.empty());
  sparse.clear();
}

TEST(array_static_convertibility) {
  using A0 = array_of_rank<int, 0>;
  using A3 = array_of_rank<int, 3>;

  static_assert(std::is_convertible<A0&, int&>::value,
                "rank-0 array should be convertible to scalar element");
  static_assert(std::is_convertible<const A0&, const int&>::value,
                "rank-0 array should be convertible to scalar element");
  static_assert(!std::is_convertible<A3&, int&>::value,
                "rank-3 array should not be convertible to element");
  static_assert(!std::is_convertible<const A3&, const int&>::value,
                "rank-3 array should not be convertible to element");
}

TEST(array_fill_constructor) {
  dense_array<int, 1> a({10}, 3);
  for (int x = 0; x < 10; x++) {
    ASSERT_EQ(a(x), 3);
  }

  dense_array<int, 2> b({7, 3}, 5);
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 7; x++) {
      ASSERT_EQ(b(x, y), 5);
    }
  }

  dense_array<int, 3> c({5, 9, 3}, 7);
  for (int z = 0; z < 3; z++) {
    for (int y = 0; y < 9; y++) {
      for (int x = 0; x < 5; x++) {
        ASSERT_EQ(c(x, y, z), 7);
      }
    }
  }

  array<int, shape<dim<>, dim<>>> sparse({{-2, 5, 2}, {4, 10, 20}}, 13);
  for (int y = 4; y < 14; y++) {
    for (int x = -2; x < 3; x++) {
      ASSERT_EQ(sparse(x, y), 13);
    }
  }
}

TEST(array_assign) {
  array_of_rank<int, 3> a({4, 5, 6});
  fill_pattern(a);

  array_of_rank<int, 3> b = array_of_rank<int, 3>({4, 5, 6});
  fill_pattern(b);
  ASSERT(a == b);

  array_of_rank<int, 3> c;
  c = array_of_rank<int, 3>({4, 5, 6});
  fill_pattern(c);
  ASSERT(a == c);

  c = array_of_rank<int, 3>({4, 5, 6});
  ASSERT(a != c);
  fill_pattern(c);
  ASSERT(a == c);

  c = array_of_rank<int, 3>({7, 5, 6});
  ASSERT(a != c);

  {
    array_of_rank<int, 3> d({4, 5, 6});
    fill_pattern(d);
    c = d;
  }
  ASSERT(a == c);
}

TEST(array_fill_assign) {
  dense_array<int, 1> a;
  a.assign({10}, 3);
  for (int x = 0; x < 10; x++) {
    ASSERT_EQ(a(x), 3);
  }

  dense_array<int, 2> b;
  b.assign({7, 3}, 5);
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 7; x++) {
      ASSERT_EQ(b(x, y), 5);
    }
  }

  dense_array<int, 3> c;
  c.assign({5, 9, 3}, 7);
  for (int z = 0; z < 3; z++) {
    for (int y = 0; y < 9; y++) {
      for (int x = 0; x < 5; x++) {
        ASSERT_EQ(c(x, y, z), 7);
      }
    }
  }

  array<int, shape<dim<>, dim<>>> sparse;
  auto sparse_shape = make_shape(dim<>(-2, 5, 2), dim<>(4, 10));
  ASSERT_LT(sparse_shape.size(), sparse_shape.flat_extent());

  sparse.assign(sparse_shape, 13);
  for (int y = 4; y < 14; y++) {
    for (int x = -2; x < 3; x++) {
      ASSERT_EQ(sparse(x, y), 13);
    }
  }
}

TEST(array_default_constructor_scalar) {
  array_of_rank<int, 0> a;
  ASSERT_EQ(a(), 0);
}

TEST(array_fill_constructor_scalar) {
  array_of_rank<int, 0> s({}, 2);
  ASSERT_EQ(s(), 2);
}

TEST(array_assign_scalar) {
  dense_array<int, 0> a({}, 3);

  dense_array<int, 0> b = array_of_rank<int, 0>({}, 3);
  ASSERT(a == b);

  array_of_rank<int, 0> c;
  c = array_of_rank<int, 0>({}, 3);
  ASSERT(a == c);

  c = array_of_rank<int, 0>({}, 4);
  ASSERT(a != c);
  c() = 3;
  ASSERT(a == c);

  {
    array_of_rank<int, 0> d({}, 3);
    c = d;
  }
  ASSERT(a == c);
}

TEST(array_fill_assign_scalar) {
  array_of_rank<int, 0> a;
  a.assign({}, 3);
  ASSERT_EQ(a(), 3);
}

TEST(sparse_array) {
  auto sparse_shape = make_shape(dim<>(-2, 5, 2), dim<>(4, 10));
  sparse_shape.resolve();
  ASSERT_LT(sparse_shape.size(), sparse_shape.flat_extent());

  array<int, shape<dim<>, dim<>>> sparse(sparse_shape);
  // Fill the storage with a constant.
  for (size_t i = 0; i < sparse_shape.flat_extent(); i++) {
    sparse.data()[i] = 7;
  }
  // Assign a different constant.
  sparse.assign(sparse_shape, 3);

  // Check that we assigned all of the elements of the array.
  for (int y = 4; y < 14; y++) {
    for (int x = -2; x < 3; x++) {
      ASSERT_EQ(sparse(x, y), 3);
    }
  }

  // Check that only the elements of the array were assigned.
  size_t sevens = 0;
  for (size_t i = 0; i < sparse_shape.flat_extent(); i++) {
    if (sparse.data()[i] == 7) { sevens++; }
  }
  ASSERT_EQ(sevens + sparse.size(), sparse_shape.flat_extent());
}

TEST(array_equality) {
  array_of_rank<int, 3> a({4, 5, 6});
  fill_pattern(a);
  array_of_rank<int, 3> b({4, 5, 6});
  fill_pattern(b);
  array_of_rank<int, 3> c({dim<>(0, 4, 2), 5, 6});
  fill_pattern(c);

  ASSERT(a == b);

  // Arrays with different strides are not equal by operator!=,
  // but equal ignores strides.
  ASSERT(a != c);
  ASSERT(equal(a, c));

  a(1, 2, 3) = 5;
  ASSERT(a != b);
  ASSERT(a != c);
  ASSERT(!equal(a, c));
}

TEST(array_copy) {
  array_of_rank<int, 3> a({4, 5, 6});
  fill_pattern(a);

  dense_array<int, 3> b({4, 5, 6});
  copy(a, b);
  check_pattern(b);

  array_of_rank<int, 3> c({dim<>(0, 4, 2), 5, 6});
  copy(b, c);
  check_pattern(c);

  array_of_rank<int, 3> d = make_copy(a, c.shape());
  ASSERT_EQ(c.shape(), d.shape());
  check_pattern(d);

  dense_array<int, 3> e = make_copy(a, b.shape());
  ASSERT_EQ(b.shape(), e.shape());
  check_pattern(e);

  dense_array<int, 3> g({{1, 2}, {1, 3}, {1, 4}});
  copy(a, g);
  check_pattern(g);

  array_of_rank<int, 3> h = make_copy(a);
  check_pattern(h);

  array_of_rank<int, 3, uninitialized_std_allocator<int>> h_uninitialized =
      make_copy(a, uninitialized_std_allocator<int>());
  check_pattern(h_uninitialized);

  array_of_rank<int, 3> i = make_copy(a.ref());
  check_pattern(i);

  array_of_rank<int, 3, uninitialized_std_allocator<int>> i_uninitialized =
      make_copy(a.ref(), uninitialized_std_allocator<int>());
  check_pattern(i_uninitialized);

  array_of_rank<int, 3> j = make_copy(a.cref());
  check_pattern(i);

  array_of_rank<int, 3, uninitialized_std_allocator<int>> j_uninitialized =
      make_copy(a.cref(), uninitialized_std_allocator<int>());
  check_pattern(j_uninitialized);
}

TEST(array_move) {
  array_of_rank<int, 3> a({4, 5, 6});
  fill_pattern(a);

  dense_array<int, 3> b({4, 5, 6});
  move(a, b);
  check_pattern(b);

  array_of_rank<int, 3> c({dim<>(0, 4, 2), 5, 6});
  move(b, c);
  check_pattern(c);

  array_of_rank<int, 3> d = make_move(a, c.shape());
  ASSERT_EQ(c.shape(), d.shape());
  check_pattern(d);

  dense_array<int, 3> e = make_move(a, b.shape());
  ASSERT_EQ(b.shape(), e.shape());
  check_pattern(e);

  dense_array<int, 3> g({4, {1, 2}, 5});
  move(a, g);
  check_pattern(g);
}

TEST(array_dense_copy) {
  array_of_rank<int, 3> source({dim<>(-3, 4, 2), 5, 6});
  fill_pattern(source);
  ASSERT(!source.is_compact());

  dense_array<int, 3> dense_copy = make_compact_copy(source);
  ASSERT(dense_copy.is_compact());
  check_pattern(dense_copy);
}

TEST(array_dense_move) {
  array_of_rank<int, 3> source({dim<>(-3, 4, 2), 5, 6});
  fill_pattern(source);
  ASSERT(!source.is_compact());

  dense_array<int, 3> dense_move = make_compact_move(source);
  ASSERT(dense_move.is_compact());
  check_pattern(dense_move);
}

TEST(array_tricky_copy) {
  array_of_rank<int, 2> source({{0, 4, 6}, {0, 6, 1}});
  fill_pattern(source);

  array_of_rank<int, 2> dest({{0, 4, 6}, {0, 3, 2}});
  copy(source, dest);
  check_pattern(dest);
}

TEST(array_for_each_value_scalar) {
  array_of_rank<int, 0> scalar;
  move_only token;
  scalar.for_each_value([token = std::move(token)](int& v) {
    v = 3;
    assert_used(token);
  });
  ASSERT_EQ(scalar(), 3);
}

TEST(array_for_each_value) {
  array_of_rank<int, 3> in_order({{0, 4, 1}, {0, 4, 4}, {0, 4, 16}});
  array_of_rank<int, 3> out_of_order({{0, 4, 16}, {0, 4, 1}, {0, 4, 4}});

  int out_of_order_counter = 0;
  move_only token;
  out_of_order.for_each_value([&, token = std::move(token)](int& v) {
    v = out_of_order_counter++;
    assert_used(token);
  });

  int in_order_counter = 0;
  in_order.for_each_value([&](int& v) { v = in_order_counter++; });

  in_order_counter = 0;
  for (int z : in_order.z()) {
    for (int y : in_order.y()) {
      for (int x : in_order.x()) {
        int expected = in_order_counter++;
        ASSERT_EQ(in_order(x, y, z), expected);
      }
    }
  }

  out_of_order_counter = 0;
  for (int x : out_of_order.x()) {
    for (int z : out_of_order.z()) {
      for (int y : out_of_order.y()) {
        int expected = out_of_order_counter++;
        ASSERT_EQ(out_of_order(x, y, z), expected);
      }
    }
  }
}

TEST(array_reshape_1d) {
  shape_of_rank<1> s({{-1, 9}});
  array_of_rank<int, 1> a(s);
  fill_pattern(a);

  a.reshape({{-3, 12}});
  check_pattern(a(s.x()));
  ASSERT_EQ(a.shape().flat_extent(), 12);

  a.reshape({5});
  check_pattern(a);
  ASSERT_EQ(a.shape().flat_extent(), 5);
}

TEST(array_reshape_3d) {
  shape_of_rank<3> s({{-1, 10}, {-2, 10}, {-3, 10}});
  array_of_rank<int, 3> a(s);
  fill_pattern(a);

  a.reshape({{-2, 12}, {-3, 13}, {-4, 14}});
  check_pattern(a(s.x(), s.y(), s.z()));
  ASSERT_EQ(a.shape().flat_extent(), 12 * 13 * 14);

  a.reshape({5, 5, 5});
  check_pattern(a);
  ASSERT_EQ(a.shape().flat_extent(), 5 * 5 * 5);
}

TEST(array_negative_strides) {
  // Test with a negative stride in y.
  array_of_rank<int, 2> a({{0, 10, 3}, {0, 3, -1}});
  fill_pattern(a);

  // Test with a negative stride in x.
  array_of_rank<int, 2> b({{0, 20, -1}, {0, 10}});
  ASSERT_EQ(b.y().stride(), 20);
  fill_pattern(b);

  // Check that making a copy with positive strides preserves
  // the contents of the array.
  dense_array<int, 2> a_compact = make_compact_copy(a);
  ASSERT_LT(0, a_compact.y().stride());
  check_pattern(a_compact);

  dense_array<int, 2> b_compact = make_compact_copy(b);
  ASSERT_EQ(b_compact.x().stride(), 1);
  check_pattern(b_compact);
}

TEST(array_move_reinterpret_shape) {
  array_of_rank<int, 3> a({9, {0, 5, 10}, 1});
  fill_pattern(a);

  auto b = move_reinterpret_shape<shape_of_rank<2>>(std::move(a));
  check_pattern(b);

  a = move_reinterpret_shape<shape_of_rank<3>>(std::move(b));
  check_pattern(a);
}

} // namespace nda

#include "array.h"
#include "test.h"

namespace array {

TEST(array_default_constructor) {
  dense_array<int, 1> a(make_dense_shape(10));
  for (int x = 0; x < 10; x++) {
    ASSERT_EQ(a(x), 0);
  }

  dense_array<int, 2> b({7, 3});
  for (int y = 0; y < 3; y++) {
    for (int x = 0; x < 7; x++) {
      ASSERT_EQ(b(x, y), 0);
    }
  }

  dense_array<int, 3> c({5, 9, 3});
  for (int z = 0; z < 3; z++) {
    for (int y = 0; y < 9; y++) {
      for (int x = 0; x < 5; x++) {
        ASSERT_EQ(c(x, y, z), 0);
      }
    }
  }

  auto sparse_shape = make_shape(dim<>(-2, 5, 2), dim<>(4, 10, 20));
  array<int, shape<dim<>, dim<>>> sparse(sparse_shape);
  for (int y = 4; y < 14; y++) {
    for (int x = -2; x < 3; x++) {
      ASSERT_EQ(sparse(x, y), 0);
    }
  }

  sparse.clear();
  ASSERT(sparse.empty());
  sparse.clear();
}

TEST(array_fill_constructor) {
  dense_array<int, 1> a(make_dense_shape(10), 3);
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

  auto sparse_shape = make_shape(dim<>(-2, 5, 2), dim<>(4, 10, 20));
  array<int, shape<dim<>, dim<>>> sparse(sparse_shape, 13);
  for (int y = 4; y < 14; y++) {
    for (int x = -2; x < 3; x++) {
      ASSERT_EQ(sparse(x, y), 13);
    }
  }
}

TEST(array_fill_assign) {
  dense_array<int, 1> a;
  a.assign(make_dense_shape(10), 3);
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
  ASSERT(sparse_shape.flat_extent() > sparse_shape.size());

  sparse.assign(sparse_shape, 13);
  for (int y = 4; y < 14; y++) {
    for (int x = -2; x < 3; x++) {
      ASSERT_EQ(sparse(x, y), 13);
    }
  }
}

TEST(sparse_array) {
  auto sparse_shape = make_shape(dim<>(-2, 5, 2), dim<>(4, 10));
  ASSERT(sparse_shape.flat_extent() > sparse_shape.size());

  array<int, shape<dim<>, dim<>>> sparse(sparse_shape);
  // Fill the storage with a constant.
  for (int i = 0; i < sparse_shape.flat_extent(); i++) {
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
  int sevens = 0;
  for (int i = 0; i < sparse_shape.flat_extent(); i++) {
    if (sparse.data()[i] == 7) {
      sevens++;
    }
  }
  ASSERT_EQ(sevens, sparse_shape.flat_extent() - sparse.size());
}

TEST(array_equality) {
  array_of_rank<int, 3> a({4, 5, 6}, 7);
  array_of_rank<int, 3> b({4, 5, 6}, 7);
  // A sparse array with a different stride than the above should still be equal to the above.
  array_of_rank<int, 3> c({dim<>(0, 4, 2), 5, 6}, 7);

  ASSERT(a == b);
  ASSERT(a == c);

  a(1, 2, 3) = 5;
  ASSERT(a != b);
  ASSERT(a != c);
}

struct lifetime_counter {
  static int default_constructs;
  static int copy_constructs;
  static int move_constructs;
  static int copy_assigns;
  static int move_assigns;
  static int destructs;

  static void reset() {
    default_constructs = 0;
    copy_constructs = 0;
    move_constructs = 0;
    copy_assigns = 0;
    move_assigns = 0;
    destructs = 0;
  }

  static int constructs() {
    return default_constructs + copy_constructs + move_constructs;
  }

  static int assigns() {
    return copy_assigns + move_assigns;
  }

  static int copies() {
    return copy_constructs + copy_assigns;
  }

  static int moves() {
    return move_constructs + move_assigns;
  }

  lifetime_counter() { default_constructs++; }
  lifetime_counter(const lifetime_counter&) { copy_constructs++; }
  lifetime_counter(lifetime_counter&&) { move_constructs++; }
  ~lifetime_counter() { destructs++; }

  lifetime_counter& operator=(const lifetime_counter&) { copy_assigns++; return *this; }
  lifetime_counter& operator=(lifetime_counter&&) { move_assigns++; return *this; }
};

int lifetime_counter::default_constructs = 0;
int lifetime_counter::copy_constructs = 0;
int lifetime_counter::move_constructs = 0;
int lifetime_counter::copy_assigns = 0;
int lifetime_counter::move_assigns = 0;
int lifetime_counter::destructs = 0;

typedef shape<dim<>, dim<>> LifetimeShape;
auto lifetime_shape = make_shape(dim<>(-2, 5, 2), dim<>(4, 10, 20));

typedef array<lifetime_counter, LifetimeShape> lifetime_array;

TEST(array_default_init_lifetime) {
  lifetime_counter::reset();
  {
    lifetime_array default_init(lifetime_shape);
  }
  ASSERT_EQ(lifetime_counter::default_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_copy_init_lifetime) {
  lifetime_counter::reset();
  {
    lifetime_array copy_init(lifetime_shape, lifetime_counter());
  }
  ASSERT_EQ(lifetime_counter::copy_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size() + 1);
}

TEST(array_copy_lifetime) {
  lifetime_array source(lifetime_shape);
  lifetime_counter::reset();
  {
    lifetime_array copy(source);
  }
  ASSERT_EQ(lifetime_counter::copy_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_move_lifetime) {
  {
    lifetime_array source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array move(std::move(source));
  }
  // This should have moved the whole array.
  ASSERT_EQ(lifetime_counter::constructs(), 0);
  ASSERT_EQ(lifetime_counter::moves(), 0);
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_move_alloc_lifetime) {
  {
    lifetime_array source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array move(std::move(source), std::allocator<lifetime_counter>());
  }
  // This should have moved the whole array.
  ASSERT_EQ(lifetime_counter::constructs(), 0);
  ASSERT_EQ(lifetime_counter::moves(), 0);
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

// TODO: Test move with incompatible allocator.

TEST(array_copy_assign_lifetime) {
  lifetime_array source(lifetime_shape);
  lifetime_counter::reset();
  {
    lifetime_array assign;
    assign = source;
  }
  ASSERT_EQ(lifetime_counter::copy_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_move_assign_lifetime) {
  {
    lifetime_array source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array assign;
    assign = std::move(source);
  }
  // This should have moved the whole array.
  ASSERT_EQ(lifetime_counter::constructs(), 0);
  ASSERT_EQ(lifetime_counter::moves(), 0);
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

// TODO: Test move with incompatible allocator.

TEST(array_clear_lifetime) {
  lifetime_counter::reset();
  lifetime_array default_init(lifetime_shape);
  default_init.clear();
  ASSERT_EQ(lifetime_counter::default_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

}  // namespace array

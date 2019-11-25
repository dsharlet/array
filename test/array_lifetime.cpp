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
#include "lifetime.h"

namespace nda {

typedef shape<dim<>, dim<>> LifetimeShape;
static LifetimeShape lifetime_shape(dim<>(-2, 5, 2), dim<>(4, 10, 20));
static LifetimeShape lifetime_subshape(dim<>(-1, 4, 2), dim<>(5, 8, 20));

template <typename Alloc>
using lifetime_array = array<lifetime_counter, LifetimeShape, Alloc>;

// Run all of these tests with 3 allocators:
typedef std::allocator<lifetime_counter> std_alloc;
typedef stack_allocator<lifetime_counter, 256> stack_alloc;

class custom_alloc {
 public:
  typedef lifetime_counter value_type;

  typedef std::true_type propagate_on_container_copy_assignment;
  typedef std::true_type propagate_on_container_move_assignment;
  typedef std::true_type propagate_on_container_swap;

  lifetime_counter* allocate(size_t n) {
    return reinterpret_cast<lifetime_counter*>(malloc(n * sizeof(lifetime_counter)));
  }
  void deallocate(lifetime_counter* p, size_t) noexcept {
    free(p);
  }

  bool operator==(const custom_alloc& other) const { return true; }
  bool operator!=(const custom_alloc& other) const { return false; }
};


template <typename Alloc>
void test_default_init_lifetime() {
  lifetime_counter::reset();
  {
    lifetime_array<Alloc> default_init(lifetime_shape);
  }
  ASSERT_EQ(lifetime_counter::default_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_default_init_lifetime) {
  test_default_init_lifetime<std_alloc>();
  test_default_init_lifetime<custom_alloc>();
  test_default_init_lifetime<stack_alloc>();
}


template <typename Alloc>
void test_default_init_constant_lifetime() {
  lifetime_counter::reset();
  typedef shape<dim<0, 4>, dim<0, 5>, dim<0, 6>> ConstantShape;
  {
    array<lifetime_counter, ConstantShape, Alloc> default_constant_init;
  }
  ConstantShape constant_shape;
  ASSERT_EQ(lifetime_counter::default_constructs, constant_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, constant_shape.size());
}

TEST(array_default_init_constant_lifetime) {
  test_default_init_constant_lifetime<std_alloc>();
  test_default_init_constant_lifetime<custom_alloc>();
  test_default_init_constant_lifetime<stack_alloc>();
}


template <typename Alloc>
void test_copy_init_lifetime() {
  lifetime_counter::reset();
  {
    lifetime_array<Alloc> copy_init(lifetime_shape, lifetime_counter());
  }
  ASSERT_EQ(lifetime_counter::copy_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size() + 1);
}

TEST(array_copy_init_lifetime) {
  test_copy_init_lifetime<std_alloc>();
  test_copy_init_lifetime<custom_alloc>();
  test_copy_init_lifetime<stack_alloc>();
}


template <typename Alloc>
void test_copy_lifetime() {
  lifetime_array<Alloc> source(lifetime_shape);
  lifetime_counter::reset();
  {
    lifetime_array<Alloc> copy(source);
  }
  ASSERT_EQ(lifetime_counter::copy_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());

  lifetime_counter::reset();
  {
    lifetime_array<Alloc> dest(lifetime_subshape);
    copy(source, dest);
  }
  ASSERT_EQ(lifetime_counter::copy_assigns, lifetime_subshape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_subshape.size());
}

TEST(array_copy_lifetime) {
  test_copy_lifetime<std_alloc>();
  test_copy_lifetime<custom_alloc>();
  test_copy_lifetime<stack_alloc>();
}


template <typename Alloc>
void test_move_lifetime(bool alloc_movable = true) {
  {
    lifetime_array<Alloc> source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array<Alloc> move(std::move(source));
  }
  // This should have moved the whole array, if the allocator is movable.
  if (alloc_movable) {
    ASSERT_EQ(lifetime_counter::constructs(), 0);
    ASSERT_EQ(lifetime_counter::moves(), 0);
    ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
  } else {
    ASSERT_EQ(lifetime_counter::constructs(), lifetime_shape.size());
    ASSERT_EQ(lifetime_counter::moves(), lifetime_shape.size());
    ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size() * 2);
  }

  {
    lifetime_array<Alloc> source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array<Alloc> dest(lifetime_subshape);
    move(source, dest);
  }
  ASSERT_EQ(lifetime_counter::constructs(), lifetime_subshape.size());
  ASSERT_EQ(lifetime_counter::move_assigns, lifetime_subshape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size() + lifetime_subshape.size());
}

TEST(array_move_lifetime) {
  test_move_lifetime<std_alloc>();
  test_move_lifetime<custom_alloc>();
  test_move_lifetime<stack_alloc>(false);
}


template <typename Alloc>
void test_move_alloc_lifetime(bool alloc_movable = true) {
  {
    lifetime_array<Alloc> source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array<Alloc> move(std::move(source), Alloc());
  }
  // This should have moved the whole array if the allocator is movable.
  if (alloc_movable) {
    ASSERT_EQ(lifetime_counter::constructs(), 0);
    ASSERT_EQ(lifetime_counter::moves(), 0);
    ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
  } else {
    ASSERT_EQ(lifetime_counter::constructs(), lifetime_shape.size());
    ASSERT_EQ(lifetime_counter::moves(), lifetime_shape.size());
    ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size() * 2);
  }
}

TEST(array_move_alloc_lifetime) {
  test_move_alloc_lifetime<std_alloc>();
  test_move_alloc_lifetime<custom_alloc>();
  test_move_alloc_lifetime<stack_alloc>(false);
}


template <typename Alloc>
void test_copy_assign_lifetime() {
  lifetime_array<Alloc> source(lifetime_shape);
  lifetime_counter::reset();
  {
    lifetime_array<Alloc> assign;
    assign = source;
  }
  ASSERT_EQ(lifetime_counter::copy_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_copy_assign_lifetime) {
  test_copy_assign_lifetime<std_alloc>();
  test_copy_assign_lifetime<custom_alloc>();
  test_copy_assign_lifetime<stack_alloc>();
}


template <typename Alloc>
void test_move_assign_lifetime(bool alloc_movable = true) {
  {
    lifetime_array<Alloc> source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array<Alloc> assign;
    assign = std::move(source);
  }
  // This should have moved the whole array if the allocator is movable.
  if (alloc_movable) {
    ASSERT_EQ(lifetime_counter::constructs(), 0);
    ASSERT_EQ(lifetime_counter::moves(), 0);
    ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
  } else {
    ASSERT_EQ(lifetime_counter::constructs(), lifetime_shape.size());
    ASSERT_EQ(lifetime_counter::moves(), lifetime_shape.size());
    ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size() * 2);
  }
}

TEST(array_move_assign_lifetime) {
  test_move_assign_lifetime<std_alloc>();
  test_move_assign_lifetime<custom_alloc>();
  test_move_assign_lifetime<stack_alloc>(false);
}


template <typename Alloc>
void test_clear_lifetime() {
  lifetime_counter::reset();
  lifetime_array<Alloc> default_init(lifetime_shape);
  default_init.clear();
  ASSERT_EQ(lifetime_counter::default_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_clear_lifetime) {
  test_clear_lifetime<std_alloc>();
  test_clear_lifetime<custom_alloc>();
  test_clear_lifetime<stack_alloc>();
}


template <typename Alloc>
void test_swap_lifetime(bool alloc_movable = true) {
  lifetime_array<Alloc> a(lifetime_shape);
  lifetime_array<Alloc> b({static_cast<index_t>(lifetime_shape.size()), 1});
  lifetime_counter::reset();
  swap(a, b);
  if (alloc_movable) {
    ASSERT_EQ(lifetime_counter::constructs(), 0);
    ASSERT_EQ(lifetime_counter::assigns(), 0);
    ASSERT_EQ(lifetime_counter::copies(), 0);
    ASSERT_EQ(lifetime_counter::moves(), 0);
  } else {
    ASSERT_EQ(lifetime_counter::default_constructs, 0);
    ASSERT_EQ(lifetime_counter::copy_constructs, 0);
    ASSERT_EQ(lifetime_counter::moves(), lifetime_shape.size() * 3);
  }
}

TEST(array_swap_lifetime) {
  test_swap_lifetime<std_alloc>();
  test_swap_lifetime<custom_alloc>();
  test_swap_lifetime<stack_alloc>(false);
}


template <typename Alloc>
void test_lifetime_leaks() {
  lifetime_counter::reset();
  {
    lifetime_array<Alloc> empty;
    lifetime_array<Alloc> default_init(lifetime_shape);
    lifetime_array<Alloc> default_init2({4, 9});
    lifetime_array<Alloc> default_init3({5, 12});
    lifetime_array<Alloc> default_init4({3, 8});
    lifetime_array<Alloc> copy(default_init);
    lifetime_array<Alloc> copy2(default_init2);
    lifetime_array<Alloc> copy3(copy2);
    lifetime_array<Alloc> copy_empty(empty);
    lifetime_array<Alloc> assign_init = default_init;
    lifetime_array<Alloc> assign_init_empty = empty;
    lifetime_array<Alloc> assign;
    assign = default_init;
    assign = default_init2;
    assign = default_init2;
    assign = default_init;
    assign = default_init;
    assign = default_init2;
    assign = default_init;
    assign = default_init2;
    assign = std::move(default_init);
    assign = std::move(default_init2);
    assign = std::move(copy2);
    assign = copy3;
    assign = copy;
    assign = copy;
    copy.clear();
    assign = copy;
    lifetime_array<Alloc> assign2;
    assign2.assign(default_init3);
    assign2.assign(default_init4);
    assign2.assign(default_init4);
    assign2.assign(default_init3);
    assign2.assign(default_init4);
    assign2.assign(lifetime_shape, lifetime_counter());
    assign2.assign(std::move(default_init3));
    assign2.assign(default_init4);
    assign2.assign(std::move(default_init4));
  }
  ASSERT_EQ(lifetime_counter::destructs, lifetime_counter::constructs());
}

TEST(array_lifetime_leaks) {
  test_lifetime_leaks<std_alloc>();
  test_lifetime_leaks<custom_alloc>();
  test_lifetime_leaks<stack_alloc>();
}

}  // namespace nda

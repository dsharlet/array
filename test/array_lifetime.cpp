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

namespace array {

typedef shape<dim<>, dim<>> LifetimeShape;
auto lifetime_shape = make_shape(dim<>(-2, 5, 2), dim<>(4, 10, 20));
auto lifetime_subshape = make_shape(dim<>(-1, 4, 2), dim<>(5, 8, 20));

typedef array<lifetime_counter, LifetimeShape> lifetime_array;

TEST(array_default_init_lifetime) {
  lifetime_counter::reset();
  {
    lifetime_array default_init(lifetime_shape);
  }
  ASSERT_EQ(lifetime_counter::default_constructs, lifetime_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size());
}

TEST(array_default_init_constant_lifetime) {
  lifetime_counter::reset();
  typedef shape<dim<0, 4>, dim<0, 5>, dim<0, 6>> ConstantShape;
  {
    array<lifetime_counter, ConstantShape> default_constant_init;
  }
  ConstantShape constant_shape;
  ASSERT_EQ(lifetime_counter::default_constructs, constant_shape.size());
  ASSERT_EQ(lifetime_counter::destructs, constant_shape.size());
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

  lifetime_counter::reset();
  {
    lifetime_array dest(lifetime_subshape);
    copy(source, dest);
  }
  ASSERT_EQ(lifetime_counter::copy_assigns, lifetime_subshape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_subshape.size());
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

  {
    lifetime_array source(lifetime_shape);
    lifetime_counter::reset();
    lifetime_array dest(lifetime_subshape);
    move(source, dest);
  }
  ASSERT_EQ(lifetime_counter::constructs(), lifetime_subshape.size());
  ASSERT_EQ(lifetime_counter::move_assigns, lifetime_subshape.size());
  ASSERT_EQ(lifetime_counter::destructs, lifetime_shape.size() + lifetime_subshape.size());
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

TEST(array_lifetime_leaks) {
  lifetime_counter::reset();
  {
    lifetime_array empty;
    lifetime_array default_init(lifetime_shape);
    lifetime_array default_init2({4, 9});
    lifetime_array copy(default_init);
    lifetime_array copy2(default_init2);
    lifetime_array copy3(copy2);
    lifetime_array copy_empty(empty);
    lifetime_array assign_init = default_init;
    lifetime_array assign_init_empty = empty;
    lifetime_array assign;
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
    assign = std::move(default_init);
    assign = std::move(default_init2);
    assign = std::move(copy2);
    assign = copy3;
    assign = copy;
    assign = copy;
    copy.clear();
    assign = copy;
    lifetime_array assign2;
    assign2.assign(default_init);
    assign2.assign(default_init);
    assign2.assign(default_init2);
    assign2.assign(default_init);
    assign2.assign(default_init2);
    assign2.assign(lifetime_shape, lifetime_counter());
    assign2.assign(std::move(default_init));
    assign2.assign(std::move(default_init2));
    assign2.assign(std::move(default_init));
    assign2.assign(std::move(default_init));
  }
  ASSERT_EQ(lifetime_counter::destructs, lifetime_counter::constructs());
}

}  // namespace array

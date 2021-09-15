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

#include <complex>
#include <cstdint>

namespace nda {

TEST(faq_c_array) {
  // Q: How do I declare an `array` with the same memory layout as a C
  // multidimensional array?

  // A: The ordering of the dimensions in memory is reversed relative to
  // the declaration order.

  // To demonstrate this, we can construct a 3-dimensional array in C,
  // where each element of the array is equal to its indices:
  constexpr int width = 5;
  constexpr int height = 4;
  constexpr int depth = 3;
  std::tuple<int, int, int> c_array[depth][height][width];
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        c_array[z][y][x] = std::make_tuple(x, y, z);
      }
    }
  }

  // Now we can create an array_ref of this c_array, and check that
  // the coordinates map to the same location:
  dense_array_ref<std::tuple<int, int, int>, 3> c_array_ref(&c_array[0][0][0], {width, height, depth});
  for_each_index(c_array_ref.shape(), [&](std::tuple<int, int, int> i) {
    ASSERT_EQ(c_array_ref[i], i);
  });

  // Q: That array generates less efficient code than my C array!

  // A: To match the generated code from C, we need to give the
  // compiler more information, like the C array declaration does.

  // This C array is equivalent to the following shape, fully known at
  // compile time:
  using CArrayShape =
      shape<dense_dim<0, width>, dim<0, height, width>, dim<0, depth, width*height>>;

  // Now we can create an array_ref of this C array, and check that
  // the coordinates map to the same location:
  array_ref<std::tuple<int, int, int>, CArrayShape> c_array_ref_fixed(&c_array[0][0][0]);
  for_each_index(c_array_ref_fixed.shape(), [&](std::tuple<int, int, int> i) {
    ASSERT_EQ(c_array_ref_fixed[i], i);
  });
}

TEST(faq_reshape) {
  // Q: How do I resize or change the shape of an already constructed array?

  // A: There are several options, with different behaviors. First, we can use
  // `array::reshape`, which changes the shape of an array while moving the
  // elements of the intersection of the old shape and new shape to the new
  // array:
  dense_array<int, 2> a({3, 4});
  for_all_indices(a.shape(), [&](int x, int y) {
    a(x, y) = y * 3 + x;
  });
  for (auto y : a.y()) {
    for (auto x : a.x()) {
      std::cout << a(x, y) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  // Output:
  // 0 1 2
  // 3 4 5
  // 6 7 8
  // 9 10 11

  a.reshape({2, 6});
  for (auto y : a.y()) {
    for (auto x : a.x()) {
      std::cout << a(x, y) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  // Output:
  // 0 1
  // 3 4
  // 6 7
  // 9 10
  // 0 0
  // 0 0

  // A: We can also reinterpret the shape of an existing array using `array::set_shape`:
  a.set_shape({4, 3});
  for (auto y : a.y()) {
    for (auto x : a.x()) {
      std::cout << a(x, y) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  // Output:
  // 0 1 3 4
  // 6 7 9 10
  // 0 0 0 0

  // A: We can also use `reinterpret_shape` to make a new `array_ref` with the
  // new shape:
  auto a_reshaped = reinterpret_shape(a, dense_shape<2>{4, 3});
  ASSERT(a_reshaped == a);

  // Q: `array`'s move constructor requires the source array to have the same
  // shape type. How do I move ownership to an array of a different shape type?

  // A: The helper function `move_reinterpret_shape` does this:
  array_of_rank<int, 3> source({3, 4, 5});
  dense_array<int, 3> dest = move_reinterpret_shape<dense_shape<3>>(std::move(source));

  // This can fail at runtime if the source shape is not compatible with the
  // destination shape.
}

TEST(faq_crop) {
  // Q: Cropping in this library is weird. How do I crop an array the way
  // (my favorite library) does it?

  // A: After cropping, the resulting array will have a min corresponding
  // to the cropped region:
  dense_array<int, 1> array({100});
  for (int i = 0; i < array.size(); i++) {
    array(i) = i;
  }
  const int crop_begin = 25;
  const int crop_end = 50;
  auto cropped = array(r(crop_begin, crop_end));
  for (int i = crop_begin; i < crop_end; i++) {
    ASSERT_EQ(array(i), cropped(i));
  }

  // This differs from most alternative libraries. To match this behavior,
  // the min of the resulting cropped array needs to be changed to 0:
  cropped.shape().dim<0>().set_min(0);
  for (int i = crop_begin; i < crop_end; i++) {
    ASSERT_EQ(array(i), cropped(i - crop_begin));
  }

  // The reason array works this way is to enable transparent tiling
  // optimizations of algorithms.
}

TEST(faq_stack_allocation) {
  // Q: How do I allocate an array on the stack?

  // A: Use `auto_allocator<>` as the allocator for your arrays.

  // Define an allocator that has storage for up to 100 elements.
  using StackAllocator = auto_allocator<int, 100>;
  ASSERT(sizeof(StackAllocator) > sizeof(int) * 100);
  dense_array<int, 3, StackAllocator> stack_array({2, 5, 10});
  // Check that the data in the array is the same as the address of the
  // array itself.
  ASSERT_EQ(stack_array.base(), reinterpret_cast<int*>(&stack_array));

  // Q: My array still isn't being allocated on the stack! Why not?

  // A: If the array is too big to fit in the allocator, it will use the
  // `BaseAlloc` of `auto_allocator`, which is `std::allocator` by default:
  dense_array<int, 3, StackAllocator> not_stack_array({3, 5, 10});
  ASSERT(not_stack_array.base() != reinterpret_cast<int*>(&not_stack_array));
}

TEST(faq_no_initialization) {
  // Q: When I declare an `array`, the memory is being initialized. I'd rather
  // not incur the cost of initialization, how can I avoid this?

  // A: Use `uninitialized_std_allocator<>` as the allocator for your arrays.

  // Define an allocator that will not default construct the values.
  using UninitialiezdAllocator = uninitialized_std_allocator<int>;
  dense_array<int, 3, UninitialiezdAllocator> stack_array({2, 5, 10});
}

} // namespace nda

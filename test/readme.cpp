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

namespace nda {

// TODO: Find a way to embed these snippets in README.md without having
// to copy-paste them.

// Define a compile-time "chunky" imag eshape.
template <int Channels, int ChannelStride = Channels>
using chunky_image_shape = shape<
    strided_dim</*Stride=*/ChannelStride>,
    dim<>,
    dense_dim</*Min=*/0, /*Extent=*/Channels>>;

// Define a compile-time small matrix type, with the array data in
// automatic storage.
template <int M, int N>
using small_matrix_shape = shape<
    dim<0, M>,
    dense_dim<0, N>>;
template <typename T, int M, int N>
using small_matrix = array<T, small_matrix_shape<M, N>, auto_allocator<T, M*N>>;
small_matrix<float, 4, 4> my_small_matrix;
// my_small_matrix is only one fixed size allocation, no dynamic allocations
// happen. sizeof(small_matrix) = sizeof(float) * 4 * 4 + (overhead)

TEST(readme) {
  // Define a 3 dimensional shape, and an array of this shape.
  using my_3d_shape_type = shape<dim<>, dim<>, dim<>>;
  constexpr int width = 16;
  constexpr int height = 10;
  constexpr int depth = 3;
  my_3d_shape_type my_3d_shape(width, height, depth);
  array<int, my_3d_shape_type> my_array(my_3d_shape);

  // Access elements of this array.
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        my_array(x, y, z) = 5;
      }
    }
  }

  // Use for_each_value helper to access this array.
  my_array.for_each_value([](int& value) {
    value = 5;
  });

  // Use for_all_indices/for_each_index helper to access this array.
  for_all_indices(my_3d_shape, [&](int x, int y, int z) {
    my_array(x, y, z) = 5;
  });
  for_each_index(my_3d_shape, [&](my_3d_shape_type::index_type i) {
    my_array[i] = 5;
  });

  // This shows the default iteration order of for_all_indices.
  my_3d_shape_type my_shape(2, 2, 2);
  for_all_indices(my_shape, [](int x, int y, int z) {
    std::cout << x << ", " << y << ", " << z << std::endl;
  });
  // Output:
  // 0, 0, 0
  // 1, 0, 0
  // 0, 1, 0
  // 1, 1, 0
  // 0, 0, 1
  // 1, 0, 1
  // 0, 1, 1
  // 1, 1, 1

  // Define a compile-time dense 3 dimensional shape.
  using my_dense_3d_shape_type = shape<
      dim</*Min=*/UNK, /*Extent=*/UNK, /*Stride=*/1>,
      dim<>,
      dim<>>;
  array<char, my_dense_3d_shape_type> my_dense_array({16, 3, 3});
  for (auto x : my_dense_array.x()) {
    // The compiler knows that each loop iteration accesses
    // elements that are contiguous in memory for contiguous x.
    my_dense_array(x, 1, 2) = 0;
  }

  // Define a matrix type with row, column indices.
  using matrix_shape = shape<dim<>, dense_dim<>>;
  array<double, matrix_shape> my_matrix({10, 4});
  for (auto i : my_matrix.i()) {
    for (auto j : my_matrix.j()) {
      // This loop ordering is efficient for this type.
      my_matrix(i, j) = 0.0;
    }
  }

  // Demonstrate slicing an array.
  // Slicing
  array_ref_of_rank<int, 2> channel1 = my_array(_, _, 1);
  array_ref_of_rank<int, 1> row4_channel2 = my_array(_, 4, 2);

  // Cropping
  array_ref_of_rank<int, 3> top_left = my_array(range<>{0, 2}, range<>{0, 4}, _);
  array_ref_of_rank<int, 2> center_channel0 = my_array(range<>{1, 2}, range<>{2, 4}, 0);

  assert_used(channel1);
  assert_used(row4_channel2);
  assert_used(top_left);
  assert_used(center_channel0);

  // Demonstrate iterating over an array in tiles using split.
  constexpr index_t x_split_factor = 3;
  const index_t y_split_factor = 5;
  for (auto yo : split(my_array.y(), y_split_factor)) {
    for (auto xo : split<x_split_factor>(my_array.x())) {
      auto tile = my_array(xo, yo, _);
      for (auto x : tile.x()) {
        // The compiler knows this loop has a fixed extent x_split_factor!
        tile(x, 0, 0) = x;
      }
    }
  }
}

}  // namespace nda


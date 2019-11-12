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
#include "benchmark.h"

#include <random>
#include <iostream>

using namespace nda;

// The standard matrix notation is to refer to elements by 'row,
// column'. To make this efficient for typical programs, we're going
// to make the second dimension the dense dim. This shape has the
// option of making the size of the matrix compile-time constant via
// the template parameters.
template <index_t Rows = UNK, index_t Cols = UNK>
using matrix_shape = shape<dim<UNK, Rows>, dense_dim<UNK, Cols>>;

// A matrix or matrix_ref is an array or array_ref with Shape =
// matrix_shape.
template <typename T, index_t Rows = UNK, index_t Cols = UNK,
          typename Alloc = std::allocator<T>>
using matrix = array<T, matrix_shape<Rows, Cols>, Alloc>;
template <typename T, index_t Rows = UNK, index_t Cols = UNK>
using matrix_ref = array_ref<T, matrix_shape<Rows, Cols>>;

// Make a reference to a submatrix of 'm', of size 'rows' x 'cols',
// starting at 'row', 'col'.
template <index_t Rows = UNK, index_t Cols = UNK, typename T>
matrix_ref<T> submatrix(const matrix_ref<T>& m, index_t row, index_t col,
                        index_t rows = Rows, index_t cols = Cols) {
  assert(row >= m.i().min());
  assert(row + rows <= m.i().max() + 1);
  assert(col >= m.j().min());
  assert(col + cols <= m.j().max() + 1);
  matrix_shape<Rows, Cols> s({row, rows, m.i().stride()}, {col, cols});
  return matrix_ref<T>(&m(row, col), s);
}

// A textbook implementation of matrix multiplication.
template <typename TAB, typename TC, index_t Rows, index_t Cols>
__attribute__((noinline))
void multiply_naive(const matrix_ref<TAB>& a, const matrix_ref<TAB>& b,
                    const matrix_ref<TC, Rows, Cols>& c) {
  for (int i : c.i()) {
    for (int j : c.j()) {
      TC c_ij = 0;
      for (int k : a.j()) {
        c_ij += a(i, k) * b(k, j);
      }
      c(i, j) = c_ij;
    }
  }
}

// This implementation of matrix multiplication reorders the loops
// so the inner loop is over the columns of the result. This makes
// it possible to vectorize the inner loop.
template <typename TAB, typename TC, index_t Rows, index_t Cols>
__attribute__((noinline))
void multiply_cols_innermost(const matrix_ref<TAB>& a, const matrix_ref<TAB>& b,
                             const matrix_ref<TC, Rows, Cols>& c) {
  for (int i : c.i()) {
    for (int j : c.j()) {
      c(i, j) = 0;
    }
    for (int k : a.j()) {
      for (int j : c.j()) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

// This implementation of matrix multiplication splits the loops over
// the output matrix into chunks, and reorders the inner loops
// innermost to form tiles. This implementation should allow the compiler
// to keep all of the accumulators for the output in registers.
template <typename TAB, typename TC>
__attribute__((noinline))
void multiply_tiles_innermost(const matrix_ref<TAB>& a, const matrix_ref<TAB>& b,
                              const matrix_ref<TC>& c) {
  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr index_t tile_rows = 3;
  constexpr index_t tile_cols = 32;
  using matrix_tile =
      matrix<TC, tile_rows, tile_cols, stack_allocator<TC, tile_rows * tile_cols>>;

  for (index_t io = 0; io < c.rows(); io += tile_rows) {
    for (index_t jo = 0; jo < c.columns(); jo += tile_cols) {
      // Make a local accumulator matrix. Hopefully this is only ever
      // stored in registers.
      matrix_tile c_tile({{io, tile_rows}, {jo, tile_cols}}, 0);
      // Compute this tile of the result.
      for (int k : a.j()) {
        for (int i : c_tile.i()) {
          for (int j : c_tile.j()) {
            c_tile(i, j) += a(i, k) * b(k, j);
          }
        }
      }
      // Copy this tile to the result. This may be cropped if the output matrix
      // size does not divide the tile size.
      index_t rows = std::min(c.rows() - io, tile_rows);
      index_t cols = std::min(c.columns() - jo, tile_cols);
      copy(c_tile, submatrix(c.ref(), io, jo, rows, cols));
    }
  }
}

int main(int, const char**) {
  // Define two input matrices.
  constexpr int M = 128;
  constexpr int K = 256;
  constexpr int N = 512;
  matrix<float> a({M, K});
  matrix<float> b({K, N});

  // 'for_each_value' calls the given function with a reference to
  // each value in the array. Use this to randomly initialize the
  // matrices with random values.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  a.for_each_value([&](float& x) { x = uniform(rng); });
  b.for_each_value([&](float& x) { x = uniform(rng); });

  // Compute the result using all matrix multiply methods.
  matrix<float> c_naive({M, N});
  double naive_time = benchmark([&]() {
    multiply_naive(a.ref(), b.ref(), c_naive.ref());
  });
  std::cout << "naive time: " << naive_time * 1e3 << " ms" << std::endl;

  matrix<float> c_cols_innermost({M, N});
  double cols_innermost_time = benchmark([&]() {
    multiply_cols_innermost(a.ref(), b.ref(), c_cols_innermost.ref());
  });
  std::cout << "rows innermost time: " << cols_innermost_time * 1e3 << " ms" << std::endl;

  matrix<float> c_tiles_innermost({M, N});
  double tiles_innermost_time = benchmark([&]() {
    multiply_tiles_innermost(a.ref(), b.ref(), c_tiles_innermost.ref());
  });
  std::cout << "tiles innermost time: " << tiles_innermost_time * 1e3 << " ms" << std::endl;

  // Verify the results from all methods are equal.
  const float epsilon = 1e-4f;
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      if (std::abs(c_cols_innermost(i, j) - c_naive(i, j)) > epsilon) {
        std::cout
          << "c_cols_innermost(" << i << ", " << j << ") = " << c_cols_innermost(i, j)
          << " != c_naive(" << i << ", " << j << ") = " << c_naive(i, j) << std::endl;
        return -1;
      }
      if (std::abs(c_tiles_innermost(i, j) - c_naive(i, j)) > epsilon) {
        std::cout
          << "c_tiles_innermost(" << i << ", " << j << ") = " << c_tiles_innermost(i, j)
          << " != c_naive(" << i << ", " << j << ") = " << c_naive(i, j) << std::endl;
        return -1;
      }
    }
  }
  return 0;
}


#include "array.h"
#include "benchmark.h"

#include <random>
#include <iostream>

using namespace array;

// The standard matrix notation is to refer to elements by 'row,
// column'. To make this efficient for typical programs, we're going
// to make the second dimension the dense dim.
template <typename T>
using matrix = array<T, shape<dim<>, dense_dim<>>>;

// A textbook implementation of matrix multiplication.
template <typename T>
void multiply_naive(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  for (int i : c.i()) {
    for (int j : c.j()) {
      T c_ij = 0;
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
template <typename T>
void multiply_cols_innermost(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
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
// TODO: This is slow. It appears to currently keep the tile in registers,
// but doesn't autovectorize like the above loop.
template <typename T>
void multiply_tiles_innermost(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr int tile_M = 4;
  constexpr int tile_N = 32;
  // TODO: Add helper functions to make this less disgusting.
  int tiles_m = c.i().extent() / tile_M;
  int tiles_n = c.j().extent() / tile_N;
  for (int io = 0; io < tiles_m; io++) {
    for (int jo = 0; jo < tiles_n; jo++) {
      for (int ii = 0; ii < tile_M; ii++) {
        int i = io * tile_M + ii;
        for (int ji = 0; ji < tile_N; ji++) {
          int j = jo * tile_N + ji;
          c(i, j) = 0;
        }
      }
      for (int k : a.j()) {
        for (int ii = 0; ii < tile_M; ii++) {
          int i = io * tile_M + ii;
          for (int ji = 0; ji < tile_N; ji++) {
            int j = jo * tile_N + ji;
            c(i, j) += a(i, k) * b(k, j);
          }
        }
      }
    }
  }
}

int main(int argc, const char** argv) {
  // Generate two random input matrices.
  constexpr int M = 128;
  constexpr int K = 256;
  constexpr int N = 512;
  matrix<float> a({M, K});
  matrix<float> b({K, N});

  // 'for_each_value' calls the given function with a reference to
  // each value in the array. Use this to randomly initialize the
  // matrices.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  a.for_each_value([&](float& x) { x = uniform(rng); });
  b.for_each_value([&](float& x) { x = uniform(rng); });

  // Compute the result using all matrix multiply methods.
  matrix<float> c_naive({M, N});
  double naive_time = benchmark([&]() {
    multiply_naive(a, b, c_naive);
  });
  std::cout << "naive time: " << naive_time * 1e3 << " ms" << std::endl;

  matrix<float> c_cols_innermost({M, N});
  double cols_innermost_time = benchmark([&]() {
    multiply_cols_innermost(a, b, c_cols_innermost);
  });
  std::cout << "rows innermost time: " << cols_innermost_time * 1e3 << " ms" << std::endl;

  matrix<float> c_tiles_innermost({M, N});
  double tiles_innermost_time = benchmark([&]() {
    multiply_tiles_innermost(a, b, c_tiles_innermost);
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
        abort();
      }
      if (std::abs(c_tiles_innermost(i, j) - c_naive(i, j)) > epsilon) {
        std::cout
          << "c_tiles_innermost(" << i << ", " << j << ") = " << c_tiles_innermost(i, j)
          << " != c_naive(" << i << ", " << j << ") = " << c_naive(i, j) << std::endl;
        abort();
      }
    }
  }
}


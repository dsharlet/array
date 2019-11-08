#include "array.h"
#include "benchmark.h"

#include <random>
#include <iostream>

using namespace array;

// The standard matrix notation is to refer to elements by 'row,
// column'. To make this efficient for typical programs, we're going
// to make the second dimension the dense dim.
template <index_t Rows = UNK, index_t Cols = UNK>
using matrix_shape = shape<dim<UNK, Rows>, dense_dim<UNK, Cols>>;
template <typename T, index_t Rows = UNK, index_t Cols = UNK>
using matrix = array<T, matrix_shape<Rows, Cols>>;
template <typename T, index_t Rows = UNK, index_t Cols = UNK>
using matrix_ref = array_ref<T, matrix_shape<Rows, Cols>>;

template <index_t Rows = UNK, index_t Cols = UNK, typename T>
matrix_ref<T> submatrix(const matrix_ref<T>& m, index_t row, index_t col, index_t rows = Rows, index_t cols = Cols) {
  return matrix_ref<T>(&m(row, col),
		       matrix_shape<Rows, Cols>(dim<>(row, rows, m.i().stride()),
						dense_dim<>(col, cols)));
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

// This implementation reorders the accumulator loop outermost.
// This is suitable for use when c is small (even if a and b are
// not).
template <typename TAB, typename TC, index_t Rows, index_t Cols>
__attribute__((noinline))
void multiply_tile(const matrix_ref<TAB>& a, const matrix_ref<TAB>& b,
		   const matrix_ref<TC, Rows, Cols>& c) {
  for (int i : c.i()) {
    for (int j : c.j()) {
      c(i, j) = 0;
    }
  }
  for (int k : a.j()) {
    for (int i : c.i()) {
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
  constexpr int tile_rows = 4;
  constexpr int tile_cols = 32;
  assert(c.rows() % tile_rows == 0);
  assert(c.columns() % tile_cols == 0);
  for (index_t i = 0; i < c.rows(); i += tile_rows) {
    for (index_t j = 0; j < c.columns(); j += tile_cols) {
      auto sub_c = submatrix<tile_rows, tile_cols>(c.ref(), i, j);
      multiply_tile(a, b, sub_c);
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


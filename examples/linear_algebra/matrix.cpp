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

#include "array/matrix.h"
#include "array/ein_reduce.h"
#include "array/z_order.h"
#include "benchmark.h"

#include <functional>
#include <iostream>
#include <random>

using namespace nda;

// Make it easier to read the generated assembly for these functions.
#define NOINLINE __attribute__((noinline))

// A textbook implementation of matrix multiplication. This is very simple,
// but it is slow, primarily because of poor locality of the loads of B. The
// reduction loop is innermost.
template <typename T>
NOINLINE void multiply_reduce_cols(const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  for (index_t i : C.i()) {
    for (index_t j : C.j()) {
      C(i, j) = 0;
      for (index_t k : A.j()) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

// This implementation uses Einstein summation. This should be equivalent
// to multiply_reduce_cols.
template <typename T>
NOINLINE void multiply_ein_reduce_cols(
    const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  fill(C, static_cast<T>(0));
  enum { i = 2, j = 1, k = 0 };
  ein_reduce(ein<i, j>(C) += ein<i, k>(A) * ein<k, j>(B));
}

// Similar to the above, but written in plain C. The timing of this version
// indicates the performance overhead (if any) of the array helpers.
template <typename TAB, typename TC>
NOINLINE void multiply_ref(const TAB* A, const TAB* B, TC* C, int M, int K, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      TC sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

// This implementation moves the reduction loop between the rows and columns
// loops. This avoids the locality problem for the loads from B. This also is
// an easier loop to vectorize (it does not vectorize a reduction variable).
template <typename T>
NOINLINE void multiply_reduce_rows(const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  for (index_t i : C.i()) {
    for (index_t j : C.j()) {
      C(i, j) = 0;
    }
    for (index_t k : A.j()) {
      for (index_t j : C.j()) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

// This implementation uses Einstein summation. This should be equivalent
// to multiply_reduce_rows.
template <class T>
NOINLINE void multiply_ein_reduce_rows(
    const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  fill(C, static_cast<T>(0));
  enum { i = 2, j = 0, k = 1 };
  ein_reduce(ein<i, j>(C) += ein<i, k>(A) * ein<k, j>(B));
}

template <typename T>
NOINLINE void multiply_reduce_matrix(
    const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  for (index_t i : C.i()) {
    for (index_t j : C.j()) {
      C(i, j) = 0;
    }
  }
  for (index_t k : A.j()) {
    for (index_t i : C.i()) {
      for (index_t j : C.j()) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

// This implementation uses Einstein summation. This should be equivalent
// to multiply_reduce_matrix.
template <class T>
NOINLINE void multiply_ein_reduce_matrix(
    const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  fill(C, static_cast<T>(0));
  enum { i = 1, j = 0, k = 2 };
  ein_reduce(ein<i, j>(C) += ein<i, k>(A) * ein<k, j>(B));
}

// This implementation of matrix multiplication splits the loops over
// the output matrix into chunks, and reorders the small loops
// innermost to form tiles. This implementation should allow the compiler
// to keep all of the accumulators for the output in registers. This
// generates an inner loop that looks like:
//
//.LBB8_12:
//	 vbroadcastss	(%rsi,%rdi,4), %ymm12
//	 vmovups	-64(%r12,%r15,4), %ymm13
//	 vmovups	-32(%r12,%r15,4), %ymm14
//	 vmovups	(%r12,%r15,4), %ymm15
//	 addq	%rbx, %r15
//	 vfmadd231ps	%ymm12, %ymm13, %ymm11
//	 vfmadd231ps	%ymm12, %ymm14, %ymm10
//	 vfmadd231ps	%ymm12, %ymm15, %ymm9
//	 vbroadcastss	(%r8,%rdi,4), %ymm12
//	 vfmadd231ps	%ymm12, %ymm13, %ymm8
//	 vfmadd231ps	%ymm12, %ymm14, %ymm7
//	 vfmadd231ps	%ymm12, %ymm15, %ymm6
//	 vbroadcastss	(%r10,%rdi,4), %ymm12
//	 vfmadd231ps	%ymm12, %ymm13, %ymm5
//	 vfmadd231ps	%ymm12, %ymm14, %ymm4
//	 vfmadd231ps	%ymm12, %ymm15, %ymm3
//	 vbroadcastss	(%rdx,%rdi,4), %ymm12
//	 incq	%rdi
//	 vfmadd231ps	%ymm13, %ymm12, %ymm2
//	 vfmadd231ps	%ymm14, %ymm12, %ymm1
//	 vfmadd231ps	%ymm12, %ymm15, %ymm0
//	 cmpq	%rdi, %r13
//	 jne	.LBB8_12
//
// This appears to achieve ~70% of the peak theoretical throughput
// of my machine.
template <typename T>
NOINLINE void multiply_reduce_tiles(const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);

  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr index_t tile_rows = 4;
  constexpr index_t tile_cols = vector_size * 3;

  for (auto io : split<tile_rows>(C.i())) {
    for (auto jo : split<tile_cols>(C.j())) {
      // Make a reference to this tile of the output.
      auto C_ijo = C(io, jo);
#if 0
      // This is slow. It would likely be fast if we could use __restrict__ on
      // struct members: https://bugs.llvm.org/show_bug.cgi?id=45863.
      fill(C_ijo, static_cast<T>(0));
      for (index_t k : A.j()) {
        for (index_t i : C_ijo.i()) {
          for (index_t j : C_ijo.j()) {
            C_ijo(i, j) += A(i, k) * B(k, j);
          }
        }
      }
#else
      // Define an accumulator buffer.
      T buffer[tile_rows * tile_cols] = {0};
      auto accumulator = make_array_ref(buffer, make_compact(C_ijo.shape()));

      // Perform the matrix multiplication for this tile.
      for (index_t k : A.j()) {
        for (index_t i : C_ijo.i()) {
          for (index_t j : C_ijo.j()) {
            accumulator(i, j) += A(i, k) * B(k, j);
          }
        }
      }

      // Copy the accumulators to the output.
#if 0
      // Not sure why this is slow. It causes the accumulators in the loop above
      // to drop out of registers.
      copy(accumulator, C_ijo);
#else
      for (index_t i : C_ijo.i()) {
        for (index_t j : C_ijo.j()) {
          C_ijo(i, j) = accumulator(i, j);
        }
      }
#endif
#endif
    }
  }
}

//  With clang -O2, this generates (almost) the same fast inner loop as the above!!
// It only spills one accumulator register, and produces statistically identical
// performance.
template <typename T>
NOINLINE void multiply_ein_reduce_tiles(
    const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);

  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr index_t tile_rows = 4;
  constexpr index_t tile_cols = vector_size * 3;

  for (auto io : split<tile_rows>(C.i())) {
    for (auto jo : split<tile_cols>(C.j())) {
      // Make a reference to this tile of the output.
      auto C_ijo = C(io, jo);
      enum { i = 1, j = 0, k = 2 };
#if 0
      // This scalarizes :( It would likely be fast if LLVM implemented
      //  __restrict__: https://bugs.llvm.org/show_bug.cgi?id=45863.
      fill(C_ijo, static_cast<T>(0));
      ein_reduce(ein<i, j>(C_ijo) += ein<i, k>(A) * ein<k, j>(B));
#else
      // Define an accumulator buffer.
      T buffer[tile_rows * tile_cols] = {0};
      auto accumulator = make_array_ref(buffer, make_compact(C_ijo.shape()));

      // Perform the matrix multiplication for this tile.
      ein_reduce(ein<i, j>(accumulator) += ein<i, k>(A) * ein<k, j>(B));

      // Copy the accumulators to the output.
#if 0
      // Not sure why this is slow. It causes the accumulators in the loop above
      // to drop out of registers.
      copy(accumulator, C_ijo);
#else
      for (index_t i : C_ijo.i()) {
        for (index_t j : C_ijo.j()) {
          C_ijo(i, j) = accumulator(i, j);
        }
      }
#endif
#endif
    }
  }
}

// This is similar to the above, but:
// - It additionally splits the reduction dimension k,
// - It traverses the 3 outer loops in z-order, improving locality.
template <typename T>
NOINLINE void multiply_ein_reduce_tiles_z_order(const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);

  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr index_t tile_rows = 4;
  constexpr index_t tile_cols = vector_size * 3;
  constexpr index_t tile_k = 128;

  // TODO: It seems like z-ordering all of io, jo, ko should be best...
  // But this seems better, even without the added convenience for initializing
  // the output.
  for (auto ko : split(A.j(), tile_k)) {
    auto split_i = split<tile_rows>(C.i());
    auto split_j = split<tile_cols>(C.j());
    for_all_z_order(std::make_tuple(split_i, split_j), [&](auto io, auto jo) {
      // Make a reference to this tile of the output.
      auto C_ijo = C(io, jo);

      // Define an accumulator buffer.
      T buffer[tile_rows * tile_cols] = {0};
      auto accumulator = make_array_ref(buffer, make_compact(C_ijo.shape()));

      // Perform the matrix multiplication for this tile.
      enum { i = 1, j = 0, k = 2 };
      ein_reduce(ein<i, j>(accumulator) += ein<i, k>(A(_, ko)) * ein<k, j>(B(ko, _)));

      // Add the accumulators to the output. Note this implementation
      // requires the tile size to divide the extent of C.
      if (ko.min() == A.j().min()) {
        for (index_t i : io) {
          for (index_t j : jo) {
            C_ijo(i, j) = accumulator(i, j);
          }
        }
      } else {
        for (index_t i : io) {
          for (index_t j : jo) {
            C_ijo(i, j) += accumulator(i, j);
          }
        }
      }
    });
  }
}

float relative_error(float A, float B) { return std::abs(A - B) / std::max(A, B); }

int main(int, const char**) {
  // Define two input matrices.
  constexpr index_t M = 384;
  constexpr index_t K = 1536;
  constexpr index_t N = 384;
  matrix<float> A({M, K});
  matrix<float> B({K, N});

  // 'for_each_value' calls the given function with a reference to
  // each value in the array. Use this to randomly initialize the
  // matrices with random values.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  generate(A, [&]() { return uniform(rng); });
  generate(B, [&]() { return uniform(rng); });

  matrix<float> c_ref({M, N});
  double ref_time = benchmark([&]() { multiply_ref(A.data(), B.data(), c_ref.data(), M, K, N); });
  std::cout << "reference time: " << ref_time * 1e3 << " ms" << std::endl;

  struct version {
    const char* name;
    std::function<void(const_matrix_ref<float>, const_matrix_ref<float>, matrix_ref<float>)> fn;
  };
  version versions[] = {
      {"reduce_cols", multiply_reduce_cols<float>},
      {"ein_reduce_cols", multiply_ein_reduce_cols<float>},
      {"reduce_rows", multiply_reduce_rows<float>},
      {"ein_reduce_rows", multiply_ein_reduce_rows<float>},
      {"reduce_matrix", multiply_reduce_matrix<float>},
      {"ein_reduce_matrix", multiply_ein_reduce_matrix<float>},
      {"reduce_tiles", multiply_reduce_tiles<float>},
      {"ein_reduce_tiles", multiply_ein_reduce_tiles<float>},
      {"ein_reduce_tiles_z_order", multiply_ein_reduce_tiles_z_order<float>},
  };
  for (auto i : versions) {
    // Compute the result using all matrix multiply methods.
    matrix<float> C({M, N});
    double time = benchmark([&]() { i.fn(A.cref(), B.cref(), C.ref()); });
    std::cout << i.name << " time: " << time * 1e3 << " ms" << std::endl;

    // Verify the results from all methods are equal.
    const float tolerance = 1e-4f;
    for (index_t i = 0; i < M; i++) {
      for (index_t j = 0; j < N; j++) {
        if (relative_error(c_ref(i, j), C(i, j)) > tolerance) {
          std::cout << "c_ref(" << i << ", " << j << ") = " << c_ref(i, j) << " != C(" << i << ", "
                    << j << ") = " << C(i, j) << std::endl;
          return -1;
        }
      }
    }
  }
  return 0;
}

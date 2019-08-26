#include "array.h"
#include "test.h"

namespace array {

template <typename T>
using matrix = array<T, shape<dim<>, dense_dim<>>>;

template <typename T>
const dim<>& rows(const matrix<T>& m) { return m.shape().template dim<0>(); }
template <typename T>
const dense_dim<>& cols(const matrix<T>& m) { return m.shape().template dim<1>(); }

// Basic implementation of matrix multiplication.
template <typename T>
void multiply_naive(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  for (int i : rows(c)) {
    for (int j : cols(c)) {
      T c_ij = 0;
      for (int k : cols(a)) {
        c_ij += a(i, k) * b(k, j);
      }
      c(i, j) = c_ij;
    }
  }
}

// This implementation of matrix multiplication reorders the loops
// so the inner loop is over the columns of the result. This makes
// it possible to vectorize the inner loop.
//
template <typename T>
void multiply(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  for (int i : rows(c)) {
    for (int j : cols(c)) {
      c(i, j) = 0;
    }
    for (int k : cols(a)) {
      for (int j : cols(c)) {
        c(i, j) += a(i, k) * b(k, j);
        // This inner loop generated the following nicely vectorized,
        // unrolled code:
        //    movq    %rcx, %rdi
        //    sarq    $32, %rdi
        //    movq    %rdi, %rsi
        //    subq    %r10, %rsi
        //    addq    %r12, %rsi
        //    vmovups (%r14,%rsi,4), %ymm2
        //    vmovups 32(%r14,%rsi,4), %ymm3
        //    vmovups 64(%r14,%rsi,4), %ymm4
        //    vmovups 96(%r14,%rsi,4), %ymm5
        //    subq    %rbx, %rdi
        //    addq    %r8, %rdi
        //    vfmadd213ps     (%r9,%rdi,4), %ymm1, %ymm2
        //    vfmadd213ps     32(%r9,%rdi,4), %ymm1, %ymm3
        //    vfmadd213ps     64(%r9,%rdi,4), %ymm1, %ymm4
        //    vfmadd213ps     96(%r9,%rdi,4), %ymm1, %ymm5
        //    vmovups %ymm2, (%r9,%rdi,4)
        //    vmovups %ymm3, 32(%r9,%rdi,4)
        //    vmovups %ymm4, 64(%r9,%rdi,4)
        //    vmovups %ymm5, 96(%r9,%rdi,4)
        //    addq    %rdx, %rcx
        //    addq    $-32, %rax
      }
    }
  }
}

// TODO: It should be possible to also generate a tiled version of the
// above, where we reorder tiles of the output innermost, instead of
// just lines. This would be much better still.

TEST(matrix_multiply) {
  // Generate two random input matrices.
  constexpr int M = 10;
  constexpr int K = 20;
  constexpr int N = 30;
  matrix<float> a({M, K});
  matrix<float> b({K, N});
  a.for_each_value([](float& x) { x = rand(); });
  b.for_each_value([](float& x) { x = rand(); });

  // Compute the result using all matrix multiply methods.
  matrix<float> c_naive({M, N});
  multiply_naive(a, b, c_naive);
  
  matrix<float> c({M, N});
  multiply(a, b, c);

  // Verify the results from all methods are equal.
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      ASSERT_EQ(c(i, j), c_naive(i, j));
    }
  }
}

}  // namespace array

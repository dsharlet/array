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

#include "ein_reduce.h"
#include "matrix.h"
#include "test.h"

#include <complex>

namespace nda {

// Helpful names for dimensions we use in ein_sums.
enum { i = 0, j = 1, k = 2, l = 3 };

TEST(make_ein_sum_diag) {
  constexpr index_t N = 64;
  matrix<int, N, N> A;
  fill_pattern(A);

  // Make diag(A), the digonal of the matrix A.
  vector<int, N> a_diag = make_ein_sum<int, i>(ein<i, i>(A));

  ASSERT_EQ(a_diag.rank(), 1);
  ASSERT_EQ(a_diag.i().extent(), N);
  for (index_t i : A.i()) {
    ASSERT_EQ(a_diag(i), A(i, i));
  }
}

TEST(ein_reduce_diag) {
  constexpr index_t N = 64;
  matrix<int, N, N> A;
  fill_pattern(A);

  // Make diag(A), the diagonal of the matrix A.
  vector<int, N> a_diag;
  // This isn't a reduction!
  ein_reduce(ein<i>(a_diag) = ein<i, i>(A));

  for (index_t i : A.i()) {
    ASSERT_EQ(a_diag(i), A(i, i));
  }
}

TEST(make_ein_sum_trace) {
  constexpr index_t N = 64;
  matrix<int, N, N> A;
  fill_pattern(A);

  // Compute trace(A) = sum(diag(A))
  int tr = make_ein_sum<int>(ein<i, i>(A));

  int tr_ref = 0;
  for (index_t i : A.i()) {
    tr_ref += A(i, i);
  }
  ASSERT_EQ(tr, tr_ref);
}

TEST(make_ein_sum_dot) {
  constexpr index_t N = 64;
  vector<int, N> x;
  vector<int, N> y;
  fill_pattern(x);
  fill_pattern(y, 2);

  // Compute the dot product x.y using an ein_sum.
  int dot = make_ein_sum<int>(ein<i>(x) * ein<i>(y));

  int dot_ref = 0;
  for (index_t i : x.i()) {
    dot_ref += x(i) * y(i);
  }
  ASSERT_EQ(dot, dot_ref);
}

TEST(ein_reduce_transpose) {
  constexpr index_t M = 12;
  constexpr index_t N = 20;
  matrix<int, M, N> A;
  fill_pattern(A);

  // Transpose A using ein_reduce.
  matrix<int, N, M> AT;
  ein_reduce(ein<i, j>(AT) = ein<j, i>(A));

  for (index_t i : AT.i()) {
    for (index_t j : AT.j()) {
      ASSERT_EQ(AT(i, j), A(j, i));
    }
  }
}

TEST(make_ein_sum_transpose) {
  constexpr index_t M = 12;
  constexpr index_t N = 20;
  matrix<int, M, N> A;
  fill_pattern(A);

  // Transpose A using an ein_sum.
  auto AT = make_ein_sum<int, i, j>(ein<j, i>(A));

  ASSERT_EQ(AT.rank(), 2);
  ASSERT_EQ(AT.rows(), A.columns());
  ASSERT_EQ(AT.columns(), A.rows());
  for (index_t i : AT.i()) {
    for (index_t j : AT.j()) {
      ASSERT_EQ(AT(i, j), A(j, i));
    }
  }
}

TEST(ein_reduce_dot_offset) {
  constexpr index_t N = 40;
  vector<int, N> x;
  vector<int, N> y;
  vector<int, N> z;
  fill_pattern(x);
  fill_pattern(y, 2);
  fill_pattern(z, 6);

  // Compute the dot product (x + y).z.
  int dot = 0;
  ein_reduce(ein<>(dot) += (ein<i>(x) + ein<i>(y)) * ein<i>(z));

  int dot_ref = 0;
  for (index_t i : x.i()) {
    dot_ref += (x(i) + y(i)) * z(i);
  }
  ASSERT_EQ(dot, dot_ref);
}

// Helpers to make a Levi-Civita tensor.
constexpr int sgn(index_t i) { return i == 0 ? 0 : (i < 0 ? -1 : 1); }

// Defines the arbitrary rank Levi-Civita tensor as a constexpr function.
constexpr int epsilon() { return 1.0f; }
template <class... Ts>
constexpr int epsilon(index_t i0, Ts... is) {
  return internal::product(sgn(is - i0)...) * epsilon(is...);
}

constexpr int epsilon3(index_t i, index_t j, index_t k) { return epsilon(i, j, k); }

TEST(ein_sum_cross) {
  const int count = 10;
  matrix<int, 3, dynamic> x({{}, count}, 0);
  matrix<int, 3, dynamic> y({{}, count}, 0);
  fill_pattern(x);
  fill_pattern(y, 3);

  // Compute the cross product of an array of vectors.
  // TODO: We can't infer the output shape of this, because ein<> of a function
  // doesn't provide a shape.
  matrix<int, 3, dynamic> cross({{}, count}, 0);
  ein_reduce(ein<i, l>(cross) += ein<i, j, k>(epsilon3) * ein<j, l>(x) * ein<k, l>(y));

  ASSERT_EQ(cross.rank(), 2);
  ASSERT_EQ(cross.rows(), 3);
  ASSERT_EQ(cross.columns(), count);
  for (int l = 0; l < count; l++) {
    ASSERT_EQ(x(1, l) * y(2, l) - x(2, l) * y(1, l), cross(0, l));
    ASSERT_EQ(x(2, l) * y(0, l) - x(0, l) * y(2, l), cross(1, l));
    ASSERT_EQ(x(0, l) * y(1, l) - x(1, l) * y(0, l), cross(2, l));
  }
}

TEST(make_ein_sum_outer) {
  constexpr index_t N = 64;
  constexpr index_t M = 40;
  vector<int, N> x;
  vector<int, M> y;
  fill_pattern(x);
  fill_pattern(y, 8);

  // Compute the outer product x^T*y.
  auto outer = make_ein_sum<int, i, j>(ein<i>(x) * ein<j>(y));

  ASSERT_EQ(outer.rank(), 2);
  ASSERT_EQ(outer.rows(), x.i().extent());
  ASSERT_EQ(outer.columns(), y.i().extent());
  for (index_t i : outer.i()) {
    for (index_t j : outer.j()) {
      ASSERT_EQ(outer(i, j), x(i) * y(j));
    }
  }
}

TEST(ein_reduce_outer) {
  constexpr index_t N = 64;
  constexpr index_t M = 40;
  vector<int, N> x;
  vector<int, M> y;
  fill_pattern(x);
  fill_pattern(y, 4);

  // Compute the outer product x^T*y.
  matrix<int, N, M> outer;
  ein_reduce(ein<i, j>(outer) = ein<i>(x) * ein<j>(y));

  for (index_t i : outer.i()) {
    for (index_t j : outer.j()) {
      ASSERT_EQ(outer(i, j), x(i) * y(j));
    }
  }
}

TEST(make_ein_sum_matrix_vector) {
  constexpr index_t M = 50;
  constexpr index_t N = 64;
  matrix<int, M, N> B;
  vector<int, N> x;
  fill_pattern(B);
  fill_pattern(x);

  // Compute the matrix-vector product B*x.
  vector<int, M> Bx = make_ein_sum<int, i>(ein<i, j>(B) * ein<j>(x));

  ASSERT_EQ(Bx.rank(), 1);
  ASSERT_EQ(Bx.i().extent(), B.rows());
  for (index_t i : Bx.i()) {
    int Bx_i = 0;
    for (index_t j : x.i()) {
      Bx_i += B(i, j) * x(j);
    }
    ASSERT_EQ(Bx(i), Bx_i);
  }
}

TEST(make_ein_sum_matrix_vector_integer) {
  constexpr index_t M = 50;
  constexpr index_t N = 64;
  matrix<char, M, N> B;
  vector<char, N> x;
  fill_pattern(B);
  fill_pattern(x);

  // Compute the matrix-vector product B*x.
  vector<int, M> Bx = make_ein_sum<int, i>(cast<short>(ein<i, j>(B)) * cast<short>(ein<j>(x)));

  ASSERT_EQ(Bx.rank(), 1);
  ASSERT_EQ(Bx.i().extent(), B.rows());
  for (index_t i : Bx.i()) {
    int Bx_i = 0;
    for (index_t j : x.i()) {
      Bx_i += static_cast<short>(B(i, j)) * static_cast<short>(x(j));
    }
    ASSERT_EQ(Bx(i), Bx_i);
  }
}

TEST(ein_sum_sum_3d) {
  array_of_rank<int, 3> T({4, 5, 8});
  fill_pattern(T);

  // Fully reduce T.
  int sum_ijk = 0;
  ein_reduce(ein(sum_ijk) -= ein<i, j, k>(T));

  int sum_ijk_ref = 0;
  T.for_each_value([&](int i) { sum_ijk_ref -= i; });
  ASSERT_EQ(sum_ijk, sum_ijk_ref);
}

TEST(make_ein_sum_sum_2d) {
  array_of_rank<int, 3> T({4, 5, 8});
  fill_pattern(T);

  // Reduce T along the i and k dimensions, keeping j.
  dense_array<int, 1> sum_ik = make_ein_sum<int, j>(ein<i, j, k>(T));

  ASSERT_EQ(sum_ik.rank(), 1);
  ASSERT_EQ(sum_ik.i().extent(), T.j().extent());
  for (index_t j : T.i()) {
    int sum_ik_ref = 0;
    T(_, j, _).for_each_value([&](int i) { sum_ik_ref += i; });
    ASSERT_EQ(sum_ik(j), sum_ik_ref);
  }
}

TEST(ein_reduce_max_2d) {
  array_of_rank<int, 3> T({4, 5, 8});
  fill_pattern(T);

  dense_array<int, 1> max_ik(dense_shape<1>{T.j().extent()}, std::numeric_limits<int>::min());

  // Reduce T along the i and k dimensions, using max as the reduction.
  auto r = ein<j>(max_ik);
  ein_reduce(r = max(r, ein<i, j, k>(T)));

  for (index_t j : T.i()) {
    int max_ik_ref = std::numeric_limits<int>::min();
    T(_, j, _).for_each_value([&](int i) { max_ik_ref = std::max(i, max_ik_ref); });
    ASSERT_EQ(max_ik(j), max_ik_ref);
  }
}

template <index_t N>
std::complex<float> W(float j, float k) {
  static const float pi = std::acos(-1.0f);
  static const std::complex<float> i(0, 1);
  return std::exp(-2.0f * pi * i * j * k / static_cast<float>(N));
}

TEST(ein_reduce_dft) {
  constexpr index_t N = 30;
  vector<float, N> x;
  fill_pattern(x);

  // Compute the DFT by multiplying by a function computing the DFT matrix.
  // This isn't fast, but it's a fun test of a reduction with a different
  // type than the operands.
  vector<std::complex<float>, N> X({}, 0.0f);
  ein_reduce(ein<j>(X) += ein<j, k>(W<N>) * ein<k>(x));

  const float tolerance = 1e-3f;
  for (index_t j = 0; j < N; j++) {
    std::complex<float> X_j_ref = 0.0f;
    for (index_t k = 0; k < N; k++) {
      X_j_ref += W<N>(j, k) * x(k);
    }
    ASSERT_LT(abs(X_j_ref - X(j)), tolerance);
  }
}

#if 0
// TODO: https://github.com/dsharlet/array/issues/42
TEST(ein_reduce_no_copy) {
  constexpr index_t N = 30;

  move_only token;
  auto non_copyable_f = [token = std::move(token)](int i) { return i; };
  vector<int, N> sum;
  ein_reduce(ein<i>(sum) = cast<int>(-ein<i>(std::move(non_copyable_f))));

  for (index_t i : sum.x()) {
    ASSERT_EQ(sum(i), -i);
  }
}
#endif

} // namespace nda

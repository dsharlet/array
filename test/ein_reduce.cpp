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

#include "matrix.h"
#include "ein_reduce.h"
#include "test.h"

namespace nda {

// Helpers to make a Levi-Civita tensor.
constexpr int sgn(index_t i) {
  return i == 0 ? 0 : (i < 0 ? -1 : 1);
}

// Defines the arbitrary rank Levi-Civita tensor as a constexpr function.
constexpr int epsilon() { return 1.0f; }
template <class... Ts>
constexpr int epsilon(index_t i0, Ts... is) {
  return internal::product(sgn(is - i0)...) * epsilon(is...);
}

constexpr int epsilon3(index_t i, index_t j, index_t k) { return epsilon(i, j, k); }

// Helpful names for dimensions we use in einsums.
enum { i = 0, j = 1, k = 2, l = 3 };

TEST(make_einsum_diag) {
  constexpr index_t N = 64;
  matrix<int, N, N> A;
  fill_pattern(A);

  // Make diag(A), the digonal of the matrix A.
  auto a_diag = make_einsum<int, i>(ein<i, i>(A));
  ASSERT_EQ(a_diag.rank(), 1);
  ASSERT_EQ(a_diag.size(), N);
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

TEST(make_einsum_trace) {
  constexpr index_t N = 64;
  matrix<int, N, N> A;
  fill_pattern(A);

  // Compute trace(A) = sum(diag(A))
  int tr = make_einsum<int>(ein<i, i>(A));
  int tr_ref = 0;
  for (index_t i : A.i()) {
    tr_ref += A(i, i);
  }
  ASSERT_EQ(tr, tr_ref);
}

TEST(make_einsum_dot) {
  constexpr index_t N = 64;
  vector<int, N> x;
  vector<int, N> y;
  fill_pattern(x);
  fill_pattern(y, {2});

  // Compute the dot product x.y using an einsum.
  int dot = make_einsum<int>(ein<i>(x) * ein<i>(y));
  int dot_ref = 0;
  for (index_t i : x.i()) {
    dot_ref += x(i) * y(i);
  }
  ASSERT_EQ(dot, dot_ref);
}

TEST(ein_reduce_dot_offset) {
  constexpr index_t N = 40;
  vector<int, N> x;
  vector<int, N> y;
  vector<int, N> z;
  fill_pattern(x);
  fill_pattern(y, {2});
  fill_pattern(z, {6});

  // Compute the dot product (x + y).z.
  int dot = 0;
  ein_reduce(ein<>(dot) += (ein<i>(x) + ein<i>(y)) * ein<i>(z));
  int dot_ref = 0;
  for (index_t i : x.i()) {
    dot_ref += (x(i) + y(i)) * z(i);
  }
  ASSERT_EQ(dot, dot_ref);
}

TEST(einsum_cross) {
  const int count = 10;
  matrix<int, 3, dynamic> x({{}, count}, 0);
  matrix<int, 3, dynamic> y({{}, count}, 0);
  fill_pattern(x);
  fill_pattern(y, {3, 4});

  // Compute the cross product of an array of vectors.
  // TODO: We can't infer the output shape of this, because ein<> of a function
  // doesn't provide a shape.
  matrix<int, 3, dynamic> cross({{}, count}, 0);
  einsum(ein<i, j, k>(epsilon3) * ein<j, l>(x) * ein<k, l>(y), ein<i, l>(cross));
  ASSERT_EQ(cross.rank(), 2);
  ASSERT_EQ(cross.rows(), 3);
  ASSERT_EQ(cross.columns(), count);
  for (int l = 0; l < count; l++) {
    ASSERT_EQ(x(1, l)*y(2, l) - x(2, l)*y(1, l), cross(0, l));
    ASSERT_EQ(x(2, l)*y(0, l) - x(0, l)*y(2, l), cross(1, l));
    ASSERT_EQ(x(0, l)*y(1, l) - x(1, l)*y(0, l), cross(2, l));
  }
}

TEST(make_einsum_outer) {
  constexpr index_t N = 64;
  constexpr index_t M = 40;
  vector<int, N> x;
  vector<int, M> y;
  fill_pattern(x);
  fill_pattern(y, {8});

  // Compute the outer product x^T*y.
  auto outer = make_einsum<int, i, j>(ein<i>(x) * ein<j>(y));
  ASSERT_EQ(outer.rank(), 2);
  ASSERT_EQ(outer.rows(), x.size());
  ASSERT_EQ(outer.columns(), y.size());
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
  fill_pattern(y, {4});

  // Compute the outer product x^T*y.
  matrix<int, N, M> outer;
  ein_reduce(ein<i, j>(outer) = ein<i>(x) * ein<j>(y));
  for (index_t i : outer.i()) {
    for (index_t j : outer.j()) {
      ASSERT_EQ(outer(i, j), x(i) * y(j));
    }
  }
}

TEST(make_einsum_matrix_vector) {
  constexpr index_t M = 50;
  constexpr index_t N = 64;
  matrix<int, M, N> B;
  vector<int, N> x;
  fill_pattern(B);
  fill_pattern(x);

  // Compute the matrix-vector product B*x.
  auto Bx = make_einsum<int, i>(ein<i, j>(B) * ein<j>(x));
  ASSERT_EQ(Bx.rank(), 1);
  ASSERT_EQ(Bx.size(), B.rows());
  for (index_t i : Bx.i()) {
    int Bx_i = 0;
    for (index_t j : x.i()) {
      Bx_i += B(i, j) * x(j);
    }
    ASSERT_EQ(Bx(i), Bx_i);
  }
}

TEST(einsum_sum_3d) {
  array_of_rank<int, 3> T({4, 5, 8});
  fill_pattern(T);

  // Fully reduce T.
  int sum_ijk = 0;
  einsum(ein<i, j, k>(T), ein(sum_ijk));
  int sum_ijk_ref = 0;
  T.for_each_value([&](int i) { sum_ijk_ref += i; });
  ASSERT_EQ(sum_ijk, sum_ijk_ref);

}

TEST(make_einsum_sum_2d) {
  array_of_rank<int, 3> T({4, 5, 8});
  fill_pattern(T);

  // Reduce T along the i and k dimensions, keeping j.
  auto sum_ik = make_einsum<int, j>(ein<i, j, k>(T));
  ASSERT_EQ(sum_ik.rank(), 1);
  ASSERT_EQ(sum_ik.size(), T.j().extent());
  for (index_t j : T.i()) {
    int sum_ik_ref = 0;
    T(_, j, _).for_each_value([&](int i) { sum_ik_ref += i; });
    ASSERT_EQ(sum_ik(j), sum_ik_ref);
  }
}

TEST(ein_reduce_max_2d) {
  array_of_rank<int, 3> T({4, 5, 8});
  fill_pattern(T);

  // Reduce T along the i and k dimensions, keeping j.
  auto max_ik = make_array<int>(make_shape(T.j()));
  ein_reduce(ein<j>(max_ik) = max(ein<j>(max_ik), ein<i, j, k>(T)));
  ASSERT_EQ(max_ik.rank(), 1);
  ASSERT_EQ(max_ik.size(), T.j().extent());
  for (index_t j : T.i()) {
    int max_ik_ref = std::numeric_limits<int>::min();
    T(_, j, _).for_each_value([&](int i) { max_ik_ref = std::max(i, max_ik_ref); });
    ASSERT_EQ(max_ik(j), max_ik_ref);
  }
}

}  // namespace nda


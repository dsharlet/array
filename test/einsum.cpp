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
#include "einsum.h"
#include "test.h"

#include <random>

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

TEST(einsum_trace) {
  constexpr index_t N = 64;
  matrix<int, N, N> A;
  fill_pattern(A);

  int tr = make_einsum<int>(ein<i, i>(A))();
  int tr_ref = 0;
  for (index_t i : A.i()) {
    tr_ref += A(i, i);
  }
  ASSERT_EQ(tr, tr_ref);
}

TEST(einsum_diag) {
  constexpr index_t N = 64;
  matrix<int, N, N> A;
  fill_pattern(A);

  auto a_diag = make_einsum<int, i>(ein<i, i>(A));
  for (index_t i : A.i()) {
    ASSERT_EQ(a_diag(i), A(i, i));
  }
}

TEST(einsum_dot) {
  constexpr index_t N = 64;
  vector<int, N> x;
  vector<int, N> y;
  fill_pattern(x);
  fill_pattern(y);

  int dot = make_einsum<int>(ein<i>(x), ein<i>(y))();
  int dot_ref = 0;
  for (index_t i : x.i()) {
    dot_ref += x(i) * y(i);
  }
  ASSERT_EQ(dot, dot_ref);
}

TEST(einsum_dot_sq) {
  constexpr index_t N = 64;
  vector<int, N> x;
  vector<int, N> y;
  fill_pattern(x);
  fill_pattern(y);

  int dot_sq = make_einsum<int>(ein<i>(x), ein<i>(x), ein<i>(y), ein<i>(y))();
  int dot_sq_ref = 0;
  for (index_t i : x.i()) {
    dot_sq_ref += x(i) * x(i) * y(i) * y(i);
  }
  ASSERT_EQ(dot_sq, dot_sq_ref);
}

TEST(einsum_cross) {
  const int count = 10;
  matrix<int, 3, dynamic> x({{}, count}, 0);
  matrix<int, 3, dynamic> y({{}, count}, 0);
  fill_pattern(x);
  fill_pattern(y);

  // TODO: We can't infer the output shape of this, because ein<> of a function
  // doesn't provide a shape.
  matrix<int, 3, dynamic> cross({{}, count}, 0);
  einsum(ein<i, j, k>(epsilon3), ein<j, l>(x), ein<k, l>(y), ein<i, l>(cross));
  ASSERT_EQ(cross.rank(), 2);
  ASSERT_EQ(cross.rows(), 3);
  ASSERT_EQ(cross.columns(), count);
  for (int l = 0; l < count; l++) {
    ASSERT_EQ(x(1, l)*y(2, l) - x(2, l)*y(1, l), cross(0, l));
    ASSERT_EQ(x(2, l)*y(0, l) - x(0, l)*y(2, l), cross(1, l));
    ASSERT_EQ(x(0, l)*y(1, l) - x(1, l)*y(0, l), cross(2, l));
  }
}

TEST(einsum_outer) {
  constexpr index_t N = 64;
  constexpr index_t M = 40;
  vector<int, N> x;
  vector<int, M> z;
  fill_pattern(x);
  fill_pattern(z);

  auto outer = make_einsum<int, i, j>(ein<i>(x), ein<j>(z));
  ASSERT_EQ(outer.rank(), 2);
  ASSERT_EQ(outer.rows(), x.size());
  ASSERT_EQ(outer.columns(), z.size());
  for (index_t i : outer.i()) {
    for (index_t j : outer.j()) {
      ASSERT_EQ(outer(i, j), x(i) * z(j));
    }
  }
}

TEST(einsum_matrix_vector) {
  constexpr index_t M = 50;
  constexpr index_t N = 64;
  matrix<int, M, N> B;
  vector<int, N> x;
  fill_pattern(B);
  fill_pattern(x);

  auto Bx = make_einsum<int, i>(ein<i, j>(B), ein<j>(x));
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

  int sum_ijk = make_einsum<int>(ein<i, j, k>(T))();
  int sum_ijk_ref = 0;
  T.for_each_value([&](int i) { sum_ijk_ref += i; });
  ASSERT_EQ(sum_ijk, sum_ijk_ref);

}

TEST(einsum_sum_2d) {
  array_of_rank<int, 3> T({4, 5, 8});
  fill_pattern(T);

  auto sum_ik = make_einsum<int, j>(ein<i, j, k>(T));
  ASSERT_EQ(sum_ik.rank(), 1);
  ASSERT_EQ(sum_ik.size(), T.j().extent());
  for (index_t j : T.i()) {
    int sum_ik_ref = 0;
    T(_, j, _).for_each_value([&](int i) { sum_ik_ref += i; });
    ASSERT_EQ(sum_ik(j), sum_ik_ref);
  }
}

}  // namespace nda


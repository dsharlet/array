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

#include <random>

using namespace nda;

float relative_error(float a, float b) {
  return std::abs(a - b) / std::max(a, b);
}

int main(int, const char**) {
  constexpr index_t M = 12;
  constexpr index_t N = 8;

  array_of_rank<float, 3> T({4, 5, 8});
  matrix<float, N, N> A;
  matrix<float, M, N> B;
  vector<float, N> x;
  vector<float, N> y;
  vector<float, M> z;

  // Helpful names for dimensions we use in einsums.
  enum { i = 0, j = 1, k = 2 };

  // `generate` assigns each element of the array with the
  // result of the generating function. Use this to fill the
  // arrays with random data.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  generate(A, [&]() { return uniform(rng); });
  generate(B, [&]() { return uniform(rng); });
  generate(x, [&]() { return uniform(rng); });
  generate(y, [&]() { return uniform(rng); });

  const float tolerance = 1e-6f;

  // trace(a)
  float tr = make_einsum<float>(ein<i, i>(A))();
  float tr_ref = 0.0f;
  for (index_t i : A.i()) {
    tr_ref += A(i, i);
  }
  assert(relative_error(tr, tr_ref) < tolerance);

  // diag(a)
  auto a_diag = make_einsum<float, i>(ein<i, i>(A));
  for (index_t i : A.i()) {
    assert(a_diag(i) == A(i, i));
  }

  // dot(x, y)
  float dot = make_einsum<float>(ein<i>(x), ein<i>(y))();
  float dot_ref = 0.0f;
  for (index_t i : A.i()) {
    dot_ref += x(i) * y(i);
  }
  assert(relative_error(dot, dot_ref) < tolerance);

  // x^T*z
  auto outer = make_einsum<float, i, j>(ein<i>(x), ein<j>(z));
  assert(outer.rows() == x.size());
  assert(outer.columns() == z.size());
  for (index_t i : outer.i()) {
    for (index_t j : outer.j()) {
      assert(outer(i, j) == x(i) * z(j));
    }
  }

  // B*x
  auto Bx = make_einsum<float, i>(ein<i, j>(B), ein<j>(x));
  for (index_t i : Bx.i()) {
    float Bx_i = 0.0f;
    for (index_t j : x.i()) {
      Bx_i += B(i, j) * x(j);
    }
    assert(relative_error(Bx(i), Bx_i) < tolerance);
  }

  // sum(T)
  float sumT = make_einsum<float>(ein<i, j, k>(T))();
  float sumT_ref = 0.0f;
  T.for_each_value([&](float i) { sumT_ref += i; });
  assert(relative_error(sumT, sumT_ref) < tolerance);

  // sum of the i and k dims of T
  auto sum_ik = make_einsum<float, j>(ein<i, j, k>(T));
  for (index_t j : T.j()) {
    float sum_ik_ref = 0.0f;
    T(_, j, _).for_each_value([&](float i) { sum_ik_ref += i; });
    assert(relative_error(sum_ik(j), sum_ik_ref) < tolerance);
  }

  return 0;
}


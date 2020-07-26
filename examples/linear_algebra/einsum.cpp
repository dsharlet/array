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
#include "benchmark.h"

#include <random>
#include <iostream>

using namespace nda;

enum { i = 0, j = 1, k = 2 };

const float tolerance = 1e-4f;

float relative_error(float a, float b) {
  return std::abs(a - b) / std::max(a, b);
}

int main(int, const char**) {
  constexpr index_t M = 12;
  constexpr index_t N = 8;

  matrix<float, N, N> a;
  matrix<float, M, N> b;
  vector<float, N> x;
  vector<float, N> y;
  vector<float, M> z;

  // `generate` assigns each element of the array with the
  // result of the generating function. Use this to fill the
  // arrays with random data.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  generate(a, [&]() { return uniform(rng); });
  generate(b, [&]() { return uniform(rng); });
  generate(x, [&]() { return uniform(rng); });
  generate(y, [&]() { return uniform(rng); });

  // trace(a)
  float tr = make_einsum<float>(ein<i, i>(a))();
  float tr_ref = 0.0f;
  for (index_t i : a.i()) {
    tr_ref += a(i, i);
  }
  assert(relative_error(tr, tr_ref) < tolerance);

  // diag(a)
  auto a_diag = make_einsum<float, i>(ein<i, i>(a));
  for (index_t i : a.i()) {
    assert(a_diag(i) == a(i, i));
  }

  // dot(x, y)
  float dot = make_einsum<float>(ein<i>(x), ein<i>(y))();
  float dot_ref = 0.0f;
  for (index_t i : a.i()) {
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

  return 0;
}


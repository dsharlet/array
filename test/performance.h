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

#ifndef ARRAY_TEST_PERFORMANCE_H
#define ARRAY_TEST_PERFORMANCE_H

#include <cmath>
#include <chrono>

namespace nda {

// Benchmark a call.
template <typename F>
double benchmark(F op) {
  op();

  const int max_trials = 10;
  const double min_time_s = 0.5;
  double time_per_iteration_s = 0;
  long iterations = 1;
  for (int trials = 0; trials < max_trials; trials++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < iterations; j++) {
      op();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    time_per_iteration_s = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / (iterations * 1e9);
    if (time_per_iteration_s * iterations > min_time_s) {
      break;
    }

    long next_iterations = std::ceil((min_time_s * 2) / time_per_iteration_s);
    iterations = std::min(std::max(next_iterations, iterations), iterations * 10);
  }
  return time_per_iteration_s;
}

// Tricks the compiler into not stripping away dead objects.
template <typename T>
__attribute__((noinline)) void assert_used(const T& x) {}

// Tricks the compiler into not constant folding the result of x.
template <typename T>
__attribute__((noinline)) T not_constant(T x) { return x; }

}  // namespace nda

#endif
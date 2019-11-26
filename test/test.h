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

#ifndef NDARRAY_TEST_TEST_H
#define NDARRAY_TEST_TEST_H

#include <cstdlib>
#include <cmath>
#include <chrono>
#include <sstream>
#include <functional>
#include <limits>
#include <iostream>

#include "array.h"

namespace nda {

inline void ostream_comma_separated_list(std::ostream&) {}

template <typename T>
void ostream_comma_separated_list(std::ostream& s, T item) {
  s << item;
}

template <typename T, typename... Ts>
void ostream_comma_separated_list(std::ostream& s, T item, Ts... items) {
  s << item << ", ";
  ostream_comma_separated_list(s, items...);
}

template <typename... Ts, size_t... Is>
void ostream_tuple(std::ostream& s, const std::tuple<Ts...>& t, std::index_sequence<Is...>) {
  ostream_comma_separated_list(s, std::get<Is>(t)...);
}

template <typename... Ts>
std::ostream& operator<<(std::ostream& s, const std::tuple<Ts...>& t) {
  s << "{";
  ostream_tuple(s, t, std::make_index_sequence<sizeof...(Ts)>());
  s << "}";
  return s;
}

template <index_t Min, index_t Extent, index_t Stride>
std::ostream& operator<<(std::ostream& s, const dim<Min, Extent, Stride>& d) {
  s << "<" << d.min() << ", " << d.extent() << ", " << d.stride() << ">";
  return s;
}

template <typename... Dims>
std::ostream& operator<<(std::ostream& s, const shape<Dims...>& sh) {
  return s << sh.dims();
}

inline float randf() {
  return rand() / static_cast<float>(RAND_MAX);
}

// Base class of a test callback.
class test {
public:
  test(const std::string& name, std::function<void()> fn);
};

void add_test(const std::string& name, std::function<void()> fn);

// A stream class that builds a message, the destructor throws an
// assert_failure exception if the check fails.
class assert_stream {
  std::stringstream msg_;
  bool fail_;

public:
 assert_stream(bool condition, const std::string& check) : fail_(!condition) {
    msg_ << check;
  }
  ~assert_stream() noexcept(false) {
    if (fail_)
      throw std::runtime_error(msg_.str());
  }

  template <typename T>
  assert_stream& operator << (const T& x) {
    if (fail_) {
      msg_ << x;
    }
    return *this;
  }
};

// Check if a and b are within epsilon of eachother, after normalization.
template <typename T>
bool roughly_equal(T a, T b, double epsilon = 1e-6) {
  return std::abs(a - b) < epsilon * std::max(std::max(std::abs(a), std::abs(b)), static_cast<T>(1));
}

// Make a new test object. The body of the test should follow this
// macro, e.g. TEST(equality) { ASSERT(1 == 1); }
#define TEST(name)                                                     \
  void test_##name##_body();                                           \
  static ::nda::test test_##name##_obj(#name, test_##name##_body);     \
  void test_##name##_body()

#define ASSERT(condition) assert_stream(condition, #condition)

#define ASSERT_EQ(a, b)                 \
  ASSERT(a == b)                        \
    << "\n" << #a << "=" << a            \
    << "\n" << #b << "=" << b << " "

#define ASSERT_LT(a, b)                 \
  ASSERT(a < b)                         \
    << "\n" << #a << "=" << a            \
    << "\n" << #b << "=" << b << " "

#define ASSERT_REQ(a, b, epsilon)      \
  ASSERT(roughly_equal(a, b, epsilon)) \
    << "\n" << #a << "=" << a           \
    << "\n" << #b << "=" << b           \
    << "\nepsilon=" << epsilon << " "

template <typename T, typename IndexType, size_t... Is>
T pattern_impl(const IndexType& indices, const IndexType& offset, std::index_sequence<Is...>) {
  static const index_t pattern_basis[] = { 10000, 100, 10, 7, 5, 3 };
  return static_cast<T>(internal::sum((std::get<Is>(indices) + std::get<Is>(offset)) * pattern_basis[Is]...));
}

// Generate a pattern from multi-dimensional indices that is generally
// suitable for detecting bugs in array operations.
template <typename T, typename IndexType>
T pattern(const IndexType& indices, const IndexType& offset = IndexType()) {
  return pattern_impl<T>(indices, offset, std::make_index_sequence<std::tuple_size<IndexType>::value>());
}

// Fill an array with the pattern.
template <typename T, typename Shape>
void fill_pattern(array<T, Shape>& a) {
  for_each_index(a.shape(), [&](const typename Shape::index_type& i) {
    a(i) = pattern<T>(i);
  });
}

// Check an array matches the pattern.
template <typename T, typename Shape>
void check_pattern(const array_ref<T, Shape>& a, const typename Shape::index_type& offset = typename Shape::index_type()) {
  for_each_index(a.shape(), [&](const typename Shape::index_type& i) {
    ASSERT_EQ(a(i), pattern<T>(i, offset)) << "i=" << i << ", offset=" << offset;
  });
}
template <typename T, typename Shape, typename Alloc>
void check_pattern(const array<T, Shape, Alloc>& a, const typename Shape::index_type& offset = typename Shape::index_type()) {
  check_pattern(a.ref(), offset);
}

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

    long next_iterations = static_cast<long>(std::ceil((min_time_s * 2) / time_per_iteration_s));
    iterations = std::min(std::max(next_iterations, iterations), iterations * 10);
  }
  return time_per_iteration_s;
}

// Tricks the compiler into not stripping away dead objects.
template <typename T>
__attribute__((noinline)) void assert_used(const T&) {}

// Tricks the compiler into not constant folding the result of x.
template <typename T>
__attribute__((noinline)) T not_constant(T x) { return x; }

}  // namespace nda

#endif  // NDARRAY_TEST_TEST_H

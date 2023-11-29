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

#include "array/array.h"
#include "test.h"

#include <vector>

namespace nda {

TEST(split_iterator_increment) {
  std::vector<interval<>> range;
  std::vector<interval<>> pre;
  std::vector<interval<>> post;

  auto test_split = split<3>(interval<>(0, 10));
  for (auto i : test_split) {
    range.push_back(i);
  }
  for (auto i = std::begin(test_split); i != std::end(test_split); ++i) {
    pre.push_back(*i);
  }
  for (auto i = std::begin(test_split); i != std::end(test_split); i++) {
    post.push_back(*i);
  }

  std::vector<interval<>> reference = {
      {0, 3},
      {3, 3},
      {6, 3},
      {7, 3},
  };

  ASSERT_EQ(range.size(), reference.size());
  ASSERT_EQ(pre.size(), reference.size());
  ASSERT_EQ(post.size(), reference.size());
  for (size_t i = 0; i < reference.size(); i++) {
    ASSERT_EQ(range[i], reference[i]);
    ASSERT_EQ(pre[i], reference[i]);
    ASSERT_EQ(post[i], reference[i]);
  }
}

void test_split_result_size(interval<> v, index_t factor) {
  auto result = split(v, factor);
  ASSERT_EQ(result.size(), (v.extent() + factor - 1) / factor);
}

template <index_t Factor>
void test_split_result_size(interval<> v) {
  auto result = split<Factor>(v);
  ASSERT_EQ(result.size(), (v.extent() + Factor - 1) / Factor);
}

TEST(split_result_size) {
  for (index_t extent = 1; extent < 100; ++extent) {
    interval<> v(0, extent);
    test_split_result_size(v, 1);
    test_split_result_size(v, 2);
    test_split_result_size(v, 3);
    test_split_result_size(v, 4);
    test_split_result_size(v, 5);
    test_split_result_size<1>(v);
    if (extent > 2) test_split_result_size<2>(v);
    if (extent > 3) test_split_result_size<3>(v);
    if (extent > 4) test_split_result_size<4>(v);
    if (extent > 5) test_split_result_size<5>(v);
  }
}

// Test compile-time constant splits that divide the extents of an array.
TEST(split_even_constant) {
  dense_array<int, 3> a({8, 9, 4});
  for (auto yo : split<3>(a.y())) {
    for (auto xo : split<4>(a.x())) {
      auto a_inner = a(xo, yo, _);
      // Compile-time constant splits should always be the same size.
      ASSERT_EQ(a_inner.size(), 48);
      fill_pattern(a_inner);
    }
  }
  check_pattern(a);
}

// Test compile-time constant splits that do not divide the extents of an array.
TEST(split_uneven_constant) {
  dense_array<int, 3> a({dense_dim<>(2, 8), dim<>(1, 9), 4});
  for (auto yo : split<4>(a.y())) {
    for (auto xo : split<5>(a.x())) {
      auto a_inner = a(xo, yo, _);
      // Compile-time constant splits should always be the same size.
      ASSERT_EQ(a_inner.size(), 80);
      fill_pattern(a_inner);
    }
  }
  check_pattern(a);
}

// Test splits that divide the extents of an array.
TEST(split_even_nonconstant) {
  dense_array<int, 3> a({4, 8, 9});
  size_t total_size = 0;
  for (auto zo : split(a.z(), 3)) {
    for (auto yo : split(a.y(), 4)) {
      auto a_inner = a(_, yo, zo);
      total_size += a_inner.size();
      fill_pattern(a_inner);
    }
  }
  // The total number of items in the inner splits should be equal to the size
  // of the array (no overlap among inner splits).
  ASSERT_EQ(total_size, a.size());
  check_pattern(a);
}

// Test splits that do not divide the extents of an array.
TEST(split_uneven_nonconstant) {
  dense_array<int, 3> a({dense_dim<>(2, 8), dim<>(1, 9), 4});
  size_t total_size = 0;
  for (auto zo : split(a.z(), 12)) {
    for (auto xo : split(a.x(), 5)) {
      auto a_inner = a(xo, _, zo);
      total_size += a_inner.size();
      fill_pattern(a_inner);
    }
  }
  // The total number of items in the inner splits should be equal to the size
  // of the array (no overlap among inner splits).
  ASSERT_EQ(total_size, a.size());
  check_pattern(a);
}

} // namespace nda

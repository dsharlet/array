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

#include "array/z_order.h"
#include "test.h"

namespace nda {

// Small wrapper to make writing tests below easier.
template <class Extents, class Fn>
void for_all_indices_in_z_order(const Extents& extents, const Fn& fn) {
  internal::for_each_index_in_z_order(extents, [&](const auto& i) { internal::apply(fn, i); });
}

TEST(for_each_in_z_order_1d) {
  // A 1-d z order traversal is useless, but it should work.
  dense_array<int, 1> a({12}, 0);
  size_t total_size = 0;
  for_all_indices_in_z_order(a.shape().extent(), [&](index_t x) {
    total_size++;
    a(x) = pattern<int>(std::make_tuple(x));
  });
  // The total number of items in the inner splits should be equal to the size
  // of the array (no overlap among inner splits).
  ASSERT_EQ(total_size, a.size());
  check_pattern(a);
}

TEST(for_each_in_z_order_2d) {
  for (index_t w : {1, 2, 3, 4, 5, 12, 16, 20}) {
    for (index_t h : {1, 2, 3, 4, 5, 12, 16, 20}) {
      dense_array<int, 2> a({w, h});
      size_t total_size = 0;
      for_all_indices_in_z_order(a.shape().extent(), [&](index_t x, index_t y) {
        total_size++;
        a(x, y) = pattern<int>(std::make_tuple(x, y));
      });
      // The total number of items in the inner splits should be equal to the size
      // of the array (no overlap among inner splits).
      ASSERT_EQ(total_size, a.size());
      check_pattern(a);
    }
  }
}

TEST(for_each_in_z_order_3d) {
  for (index_t w : {1, 2, 3, 4, 5, 12, 16, 20}) {
    for (index_t h : {1, 2, 3, 4, 5, 12, 16, 20}) {
      for (index_t d : {1, 2, 3, 4, 5, 12, 16, 20}) {
        dense_array<int, 3> a({w, h, d});
        size_t total_size = 0;
        for_all_indices_in_z_order(a.shape().extent(), [&](index_t x, index_t y, index_t z) {
          total_size++;
          a(x, y, z) = pattern<int>(std::make_tuple(x, y, z));
        });
        // The total number of items in the inner splits should be equal to the size
        // of the array (no overlap among inner splits).
        ASSERT_EQ(total_size, a.size());
        check_pattern(a);
      }
    }
  }
}

TEST(split_z_order_2d) {
  for (index_t w : {3, 4, 5, 6, 12, 20}) {
    for (index_t h : {1, 2, 3, 4, 5, 6, 12, 20}) {
      dense_array<int, 2> a({w, h});
      size_t total_size = 0;
      auto xs = split<3>(a.x());
      auto ys = split(a.y(), 3);
      for_all_in_z_order(std::make_tuple(xs, ys), [&](const auto& xo, const auto& yo) {
        auto a_inner = a(xo, yo);
        total_size += a_inner.size();
        fill_pattern(a_inner);
      });
      // The total number of items in the inner splits should be equal to the size
      // of the array (no overlap among inner splits).
      ASSERT_LE(a.size(), total_size);
      check_pattern(a);
    }
  }
}

TEST(split_z_order_3d) {
  for (index_t w : {2, 3, 4, 5, 12, 16, 20}) {
    for (index_t h : {1, 2, 3, 4, 5, 12, 16, 20}) {
      for (index_t d : {4, 5, 12, 16, 20}) {
        dense_array<int, 3> a({w, h, d});
        size_t total_size = 0;
        auto xs = split<2>(a.x());
        auto ys = split(a.y(), 3);
        auto zs = split<4>(a.z());
        for_all_in_z_order(std::make_tuple(xs, ys, zs), [&](const auto& xo, const auto& yo, const auto& zo) {
          auto a_inner = a(xo, yo, zo);
          total_size += a_inner.size();
          fill_pattern(a_inner);
        });
        // The total number of items in the inner splits should be equal to the size
        // of the array (no overlap among inner splits).
        ASSERT_LE(a.size(), total_size);
        check_pattern(a);
      }
    }
  }
}

} // namespace nda

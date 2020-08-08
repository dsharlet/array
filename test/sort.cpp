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

#include "array.h"
#include "test.h"

#include <algorithm>
#include <vector>

namespace nda {

TEST(sort) {
  for (int size = 0; size < 100; size++) {
    std::vector<int> unordered(size);
    for (int& i : unordered) {
      i = rand() % 100;
    }

    std::vector<int> std_sorted(unordered);
    std::vector<int> sorted(unordered);

    std::sort(std_sorted.begin(), std_sorted.end());
    internal::bubble_sort(sorted.begin(), sorted.end());

    for (size_t i = 0; i < unordered.size(); i++) {
      ASSERT_EQ(std_sorted[i], sorted[i]);
    }
  }
}

} // namespace nda

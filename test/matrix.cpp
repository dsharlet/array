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
#include "test.h"

namespace nda {

TEST(matrix_slice) {
  matrix<float> m({10, 10}, 0);
  for_all_indices(m.shape(), [&](int i, int j) {
    m(i, j) = i * 10 + j;
  });

  for (int i = 0; i < 10; i++) {
    auto row = m(_, i);
    auto col = m(i, _);

    for(int j = 0; j < 10; j++) {
      ASSERT_EQ(row(j), j * 10 + i);
      ASSERT_EQ(col(j), i * 10 + j);
    }
  }
}

}  // namespace nda

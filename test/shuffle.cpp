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

namespace nda {
namespace internal {

TEST(shuffle_1_of_3) {
  auto t = std::make_tuple(0, 1, 2);
  auto shuffled0 = shuffle<0>(t);
  auto shuffled1 = shuffle<1>(t);
  auto shuffled2 = shuffle<2>(t);
  ASSERT_EQ(shuffled0, std::make_tuple(0));
  ASSERT_EQ(shuffled1, std::make_tuple(1));
  ASSERT_EQ(shuffled2, std::make_tuple(2));

  // Does inverse_shuffle for this make sense?
}

TEST(shuffle_2_of_3) {
  auto t = std::make_tuple(0, 1, 2);
  auto shuffled01 = shuffle<0, 1>(t);
  auto shuffled10 = shuffle<1, 0>(t);
  auto shuffled12 = shuffle<1, 2>(t);
  auto shuffled21 = shuffle<2, 1>(t);
  auto shuffled02 = shuffle<0, 2>(t);
  auto shuffled20 = shuffle<2, 0>(t);
  ASSERT_EQ(shuffled01, std::make_tuple(0, 1));
  ASSERT_EQ(shuffled10, std::make_tuple(1, 0));
  ASSERT_EQ(shuffled12, std::make_tuple(1, 2));
  ASSERT_EQ(shuffled21, std::make_tuple(2, 1));
  ASSERT_EQ(shuffled02, std::make_tuple(0, 2));
  ASSERT_EQ(shuffled20, std::make_tuple(2, 0));

  // Does inverse_shuffle for this make sense?
}

TEST(shuffle_3_of_3) {
  auto t = std::make_tuple(0, 1, 2);
  auto shuffled012 = shuffle<0, 1, 2>(t);
  auto shuffled021 = shuffle<0, 2, 1>(t);
  auto shuffled102 = shuffle<1, 0, 2>(t);
  auto shuffled120 = shuffle<1, 2, 0>(t);
  auto shuffled201 = shuffle<2, 0, 1>(t);
  auto shuffled210 = shuffle<2, 1, 0>(t);
  ASSERT_EQ(shuffled012, std::make_tuple(0, 1, 2));
  ASSERT_EQ(shuffled021, std::make_tuple(0, 2, 1));
  ASSERT_EQ(shuffled102, std::make_tuple(1, 0, 2));
  ASSERT_EQ(shuffled120, std::make_tuple(1, 2, 0));
  ASSERT_EQ(shuffled201, std::make_tuple(2, 0, 1));
  ASSERT_EQ(shuffled210, std::make_tuple(2, 1, 0));

  auto t0 = inverse_shuffle<0, 1, 2>(shuffled012);
  auto t1 = inverse_shuffle<0, 2, 1>(shuffled021);
  auto t2 = inverse_shuffle<1, 0, 2>(shuffled102);
  auto t3 = inverse_shuffle<1, 2, 0>(shuffled120);
  auto t4 = inverse_shuffle<2, 0, 1>(shuffled201);
  auto t5 = inverse_shuffle<2, 1, 0>(shuffled210);
  ASSERT_EQ(t0, t);
  ASSERT_EQ(t1, t);
  ASSERT_EQ(t2, t);
  ASSERT_EQ(t3, t);
  ASSERT_EQ(t4, t);
  ASSERT_EQ(t5, t);
}

}  // namespace internal
}  // namespace nda

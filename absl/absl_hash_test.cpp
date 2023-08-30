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

#include "absl/absl.h"
#include "array/array.h"
#include "gtest/gtest.h"

#include "absl/hash/hash_testing.h"

namespace nda {

TEST(absl_hash, interval) {
  interval<4> i3;
  i3.set_extent(6);

  ASSERT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({std::make_tuple(
      interval<>(), interval<>(), interval<>(-6, 11), interval<3>(), i3, interval<-5, -12>())}));
}

TEST(absl_hash, dim) {
  dim<> d0;
  dim<> d1(/*extent=*/640);
  dim<> d2(/*min=*/35, /*extent=*/640);
  dim<> d3(/*min=*/77, /*extent=*/480, /*stride=*/2);
  dim</*min=*/3> d4;
  dim</*min=*/-4, /*extent=*/5> d5;
  dim</*min=*/10, /*extent=*/11, /*stride=*/-1> d6;

  ASSERT_TRUE(
      absl::VerifyTypeImplementsAbslHashCorrectly({std::make_tuple(d0, d1, d2, d3, d4, d5, d6)}));
}

TEST(absl_hash, shape) {
  shape_of_rank<0> sh0;
  shape_of_rank<1> sh1;
  shape_of_rank<3> sh2;
  dense_shape<2> sh3;
  sh3.dim<0>().set_extent(10);
  sh3.dim<1>().set_min(6);
  sh3.dim<1>().set_extent(2);
  sh3.dim<1>().set_stride(16);

  dense_shape<2> sh4;
  sh4.dim<0>().set_extent(10);
  sh4.dim<1>().set_min(6);
  sh4.dim<1>().set_extent(2);
  sh4.dim<1>().set_stride(16);

  fixed_dense_shape<640, 480, 3> sh5;

  dense_dim<> x(0, 10);
  dim<> y(67, 5, -x.extent()); // Dynamic values, fully specified.
  dim<> z(-11, 103);           // Dynamic values, unspecified stride.

  shape<dense_dim<>, dim<>, dim<>> sh6(x, y, z);

  shape<dense_dim<>, dim<>, dim<>> sh7(x, y, z);
  sh7.resolve();

  ASSERT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {std::make_tuple(sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7)}));
}

} // namespace nda

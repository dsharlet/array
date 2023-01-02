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
#include "array.h"
#include "test/test.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include <string>

namespace nda {

TEST(absl_stringify_interval) {
  {
    interval<> i0;
    const std::string s = absl::StrFormat("%v", i0);
    ASSERT(s == "[0, 1)");
  }

  {
    interval<> i1(-6, 11);
    const std::string s = absl::StrFormat("%v", i1);
    ASSERT(s == "[-6, 5)");
  }

  {
    interval<3> i2;
    const std::string s = absl::StrFormat("%v", i2);
    ASSERT(s == "[3, 4)");
  }

  {
    interval<4> i3;
    i3.set_extent(6);
    const std::string s = absl::StrFormat("%v", i3);
    ASSERT(s == "[4, 10)");
  }

  {
    interval<-5, -12> i4;
    const std::string s = absl::StrFormat("%v", i4);
    ASSERT(s == "[-5, -17)");
  }
}

TEST(absl_stringify_dim) {
  // Fully dynamic but unset.
  {
    dim<> d;
    const std::string s = absl::StrFormat("%v", d);
    ASSERT(s == "dim(0, 0, -9)");
  }

  {
    dim<> d(/*extent=*/640);
    const std::string s = absl::StrFormat("%v", d);
    ASSERT(s == "dim(0, 640, -9)");
  }

  {
    dim<> d(/*min=*/35, /*extent=*/640);
    const std::string s = absl::StrFormat("%v", d);
    ASSERT(s == "dim(35, 640, -9)");
  }

  {
    dim<> d(/*min=*/77, /*extent=*/480, /*stride=*/2);
    const std::string s = absl::StrFormat("%v", d);
    ASSERT(s == "dim(77, 480, 2)");
  }

  // Statically set min/extent/stride.
  {
    dim</*min=*/3> d;
    const std::string s = absl::StrFormat("%v", d);
    ASSERT(s == "dim(3, 0, -9)");
  }

  {
    dim</*min=*/-4, /*extent=*/5> d;
    const std::string s = absl::StrFormat("%v", d);
    ASSERT(s == "dim(-4, 5, -9)");
  }

  {
    dim</*min=*/10, /*extent=*/11, /*stride=*/-1> d;
    const std::string s = absl::StrFormat("%v", d);
    ASSERT(s == "dim(10, 11, -1)");
  }
}

TEST(absl_stringify_shape) {
  // Fully dynamic but unset.
  {
    shape_of_rank<0> sh;
    const std::string s = absl::StrFormat("%v", sh);
    ASSERT(s == "shape<0>()");
  }

  // Fully dynamic but unset.
  {
    shape_of_rank<1> sh;
    const std::string s = absl::StrFormat("%v", sh);
    ASSERT(s == "shape<1>(dim(0, 0, -9))");
  }

  {
    shape_of_rank<3> sh;
    const std::string s = absl::StrFormat("%v", sh);
    ASSERT(s == "shape<3>(dim(0, 0, -9), dim(0, 0, -9), dim(0, 0, -9))");
  }

  // Test a few static cases.
  {
    dense_shape<2> sh;
    const std::string s = absl::StrFormat("%v", sh);
    ASSERT(s == "shape<2>(dim(0, 0, 1), dim(0, 0, -9))");

    sh.dim<0>().set_extent(10);
    sh.dim<1>().set_min(6);
    sh.dim<1>().set_extent(2);
    sh.dim<1>().set_stride(16);
    const std::string s2 = absl::StrFormat("%v", sh);
    ASSERT(s2 == "shape<2>(dim(0, 10, 1), dim(6, 2, 16))");
  }

  {
    dense_shape<2> sh;
    const std::string s = absl::StrFormat("%v", sh);
    ASSERT(s == "shape<2>(dim(0, 0, 1), dim(0, 0, -9))");

    sh.dim<0>().set_extent(10);
    sh.dim<1>().set_min(6);
    sh.dim<1>().set_extent(2);
    sh.dim<1>().set_stride(16);
    const std::string s2 = absl::StrFormat("%v", sh);
    ASSERT(s2 == "shape<2>(dim(0, 10, 1), dim(6, 2, 16))");
  }

  {
    fixed_dense_shape<640, 480, 3> sh;
    const std::string s = absl::StrFormat("%v", sh);
    ASSERT(s == "shape<3>(dim(0, 640, 1), dim(0, 480, 640), dim(0, 3, 307200))");
  }

  // Custom dims and negative stride.
  {
    dense_dim<> x(0, 10);
    dim<> y(67, 5, -x.extent()); // Dynamic values, fully specified.
    dim<> z(-11, 103);           // Dynamic values, unspecified stride.

    shape<dense_dim<>, dim<>, dim<>> sh(x, y, z);
    const std::string s = absl::StrFormat("%v", sh);
    ASSERT(s == "shape<3>(dim(0, 10, 1), dim(67, 5, -10), dim(-11, 103, -9))");

    // Resolving the stride yields z_stride = 10 * 5 = 50.
    sh.resolve();
    const std::string s2 = absl::StrFormat("%v", sh);
    ASSERT(s2 == "shape<3>(dim(0, 10, 1), dim(67, 5, -10), dim(-11, 103, 50))");
  }
}

} // namespace nda

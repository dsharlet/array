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
  interval<> i0;
  {
    const std::string s = absl::StrFormat("%v", i0);
    ASSERT(s == "[0, 1)");
  }

  interval<> i1(-6, 11);
  {
    const std::string s = absl::StrFormat("%v", i1);
    ASSERT(s == "[-6, 5)");
  }

  interval<3> i2;
  {
    const std::string s = absl::StrFormat("%v", i2);
    ASSERT(s == "[3, 4)");
  }

  interval<4> i3;
  i3.set_extent(6);
  {
    const std::string s = absl::StrFormat("%v", i3);
    ASSERT(s == "[4, 10)");
  }

  interval<-5, -12> i4;
  {
    const std::string s = absl::StrFormat("%v", i4);
    ASSERT(s == "[-5, -17)");
  }
}

} // namespace nda

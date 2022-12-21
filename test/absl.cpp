#include "array.h"
#include "array_absl.h"
#include "test.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include <string>

namespace nda {

TEST(absl_stringify_interval) {
  interval<> i0;
  {
    const std::string s = absl::StrFormat("%v", i0);
    ASSERT(s == "[0, 0] (extent = 1)");
  }

  interval<> i1(-6, 11);
  {
    const std::string s = absl::StrFormat("%v", i1);
    ASSERT(s == "[-6, 4] (extent = 11)");
  }

  interval<3> i2;
  {
    const std::string s = absl::StrFormat("%v", i2);
    ASSERT(s == "[3, 3] (extent = 1)");
  }

  interval<4> i3;
  i3.set_extent(6);
  {
    const std::string s = absl::StrFormat("%v", i3);
    ASSERT(s == "[4, 9] (extent = 6)");
  }

  interval<-5, -12> i4;
  {
    const std::string s = absl::StrFormat("%v", i4);
    ASSERT(s == "[-5, -18] (extent = -12)");
  }
}

} // namespace nda

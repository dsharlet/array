#include "absl/strings/str_format.h"
#include "array.h"

// Adds Abseil Stringify support (https://abseil.io/blog/20221115-stringify).

namespace nda {

// interval -> string.
template <typename Sink, index_t Min_ = dynamic, index_t Extent_ = dynamic>
void AbslStringify(Sink& sink, const nda::interval<Min_, Extent_>& p) {
  absl::Format(&sink, "[%v, %v] (extent = %v)", p.min(), p.max(), p.extent());
}

} // namespace nda

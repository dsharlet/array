#include "absl/strings/str_format.h"
#include "array.h"

// Adds Abseil Stringify support (https://abseil.io/blog/20221115-stringify).

namespace nda {

// interval -> string as the half-open interval [begin, end).
//
// Stringifies only the values, not whether they are static or dynamic.
template <typename Sink, index_t Min_ = dynamic, index_t Extent_ = dynamic>
void AbslStringify(Sink& sink, const interval<Min_, Extent_>& i) {
  absl::Format(&sink, "[%v, %v)", i.min(), i.min() + i.extent());
}

// dim -> string as "dim(min, extent, stride)".
//
// Stringifies only the values, not whether they are static or dynamic.
template <typename Sink, index_t Min_ = dynamic, index_t Extent_ = dynamic,
    index_t Stride_ = dynamic>
void AbslStringify(Sink& sink, const dim<Min_, Extent_, Stride_>& d) {
  absl::Format(&sink, "dim(%v, %v, %v)", d.min(), d.extent(), d.stride());
}

// shape -> string as "shape<`rank`>(dims...).
//
// Stringifies only the rank and each dim's values, not whether they are static
// or dynamic.
//
// TODO(jiawen): This can be implemented at compile-time using index_sequence
// (and a fold expression in C++17) but is not worth the reduced readability.
template <typename Sink, class... Dims>
void AbslStringify(Sink& sink, const shape<Dims...>& sh) {
  absl::Format(&sink, "shape<%d>(", sh.rank());
  for (int i = 0; i < sh.rank(); ++i) {
    if (i == 0) {
      absl::Format(&sink, "%v", sh.dim(i));
    } else {
      absl::Format(&sink, ", %v", sh.dim(i));
    }
  }
  sink.Append(")");
}

} // namespace nda

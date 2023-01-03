#ifndef NDARRAY_BOUNDS_H
#define NDARRAY_BOUNDS_H

#include <tuple>

#include "array.h"

namespace nda {

namespace internal {

template <class IntervalDst, class IntervalSrc>
NDARRAY_HOST_DEVICE void assert_interval_compatible(size_t interval_index, const IntervalSrc& src) {
  bool compatible = true;
  if (is_static(IntervalDst::Min) && !is_dynamic(src.min()) && src.min() != IntervalDst::Min) {
    NDARRAY_PRINT_ERR("Error converting interval %zu: expected static min " NDARRAY_INDEX_T_FMT
                      ", got " NDARRAY_INDEX_T_FMT "\n",
        interval_index, IntervalDst::Min, src.min());
    compatible = false;
  }
  if (is_static(IntervalDst::Extent) && !is_dynamic(src.extent()) &&
      src.extent() != IntervalDst::Extent) {
    NDARRAY_PRINT_ERR("Error converting interval %zu: expected static extent " NDARRAY_INDEX_T_FMT
                      ", got " NDARRAY_INDEX_T_FMT "\n",
        interval_index, IntervalDst::Extent, src.extent());
    compatible = false;
  }
  assert(compatible);
  (void)compatible;
}

template <class IntervalDst, class IntervalSrc, size_t... Is>
NDARRAY_HOST_DEVICE void assert_intervals_compatible(
    const IntervalSrc& src, index_sequence<Is...>) {
  // This is ugly, in C++17, we'd use a fold expression over the comma operator (f(), ...).
  int unused[] = {(assert_interval_compatible<typename std::tuple_element<Is, IntervalDst>::type>(
                       Is, nda::interval<>(std::get<Is>(src))),
      0)...};
  (void)unused;
}

template <class IntervalDst, class IntervalSrc>
NDARRAY_HOST_DEVICE const IntervalSrc& assert_intervals_compatible(const IntervalSrc& src) {
#ifndef NDEBUG
  assert_intervals_compatible<IntervalDst>(
      src, make_index_sequence<std::tuple_size<IntervalDst>::value>());
#endif
  return src;
}

template <class... Intervals, size_t... Is>
NDARRAY_HOST_DEVICE auto dims(const std::tuple<Intervals...>& intervals, index_sequence<Is...>) {
  return std::make_tuple(dim(std::get<Is>(intervals))...);
}

} // namespace internal

template <class... Intervals>
class bounds;

/** Helper function to make a bounds from a variadic list of `intervals...`. */
template <class... Intervals>
NDARRAY_HOST_DEVICE auto make_bounds(Intervals... intervals) {
  return bounds<Intervals...>(intervals...);
}

template <class... Intervals>
NDARRAY_HOST_DEVICE bounds<Intervals...> make_bounds_from_tuple(
    const std::tuple<Intervals...>& intervals) {
  return bounds<Intervals...>(intervals);
}

template <class... Intervals>
class bounds {
public:
  /** The type of the intervals tuple of this bounds. */
  using intervals_type = std::tuple<Intervals...>;

  /** Number of intervals in this bounds. */
  static constexpr size_t rank() { return std::tuple_size<intervals_type>::value; }

  /** A bounds is scalar if its rank is 0. */
  static constexpr bool is_scalar() { return rank() == 0; }

  /** The type of an index for this bounds. */
  using index_type = index_of_rank<rank()>;

  using size_type = size_t;

  // A frequently-used alias to make the index sequence [0, 1, ..., rank()).
  using interval_indices = decltype(internal::make_index_sequence<rank()>());

private:
  intervals_type intervals_;

  // TODO: This should use std::is_constructible<intervals_type, std::tuple<OtherIntervals...>>
  // but it is broken on some compilers (https://github.com/dsharlet/array/issues/20).
  template <class... OtherIntervals>
  using enable_if_intervals_compatible = std::enable_if_t<sizeof...(OtherIntervals) == rank()>;

  template <size_t I>
  using enable_if_interval = std::enable_if_t<(I < rank())>;

public:
  NDARRAY_HOST_DEVICE bounds() {}
  // TODO: This is a bit messy, but necessary to avoid ambiguous default
  // constructors when Intervals is empty.
  template <size_t N = sizeof...(Intervals), class = std::enable_if_t<(N > 0)>>
  NDARRAY_HOST_DEVICE bounds(const Intervals&... intervals)
      : intervals_(
            internal::assert_intervals_compatible<intervals_type>(std::make_tuple(intervals...))) {}

  NDARRAY_HOST_DEVICE bounds(const bounds&) = default;
  NDARRAY_HOST_DEVICE bounds(bounds&&) = default;
  NDARRAY_HOST_DEVICE bounds& operator=(const bounds&) = default;
  NDARRAY_HOST_DEVICE bounds& operator=(bounds&&) = default;

  /** Construct or assign a bounds from another set of intervals of a possibly
   * different type. Each interval must be compatible with the corresponding
   * interval of this bounds. */
  template <class... OtherIntervals, class = enable_if_intervals_compatible<OtherIntervals...>>
  NDARRAY_HOST_DEVICE bounds(const std::tuple<OtherIntervals...>& other)
      : intervals_(internal::assert_intervals_compatible<intervals_type>(other)) {}
  // ELEPHANT - ambiguous with the second constructor?
  // template <class... OtherIntervals, class = enable_if_intervals_compatible<OtherIntervals...>>
  // NDARRAY_HOST_DEVICE bounds(OtherIntervals... other_intervals)
  //  : bounds(std::make_tuple(other_intervals...)) {}
  template <class... OtherIntervals, class = enable_if_intervals_compatible<OtherIntervals...>>
  NDARRAY_HOST_DEVICE bounds(const bounds<OtherIntervals...>& other)
      : intervals_(internal::assert_intervals_compatible<intervals_type>(other.intervals())) {}
  template <class... OtherIntervals, class = enable_if_intervals_compatible<OtherIntervals...>>
  NDARRAY_HOST_DEVICE bounds& operator=(const bounds<OtherIntervals...>& other) {
    intervals_ = internal::assert_intervals_compatible<intervals_type>(other.intervals());
    return *this;
  }

  /** Get a specific interval `I` of this bounds. */
  template <size_t I, class = enable_if_interval<I>>
  NDARRAY_HOST_DEVICE auto& interval() {
    return std::get<I>(intervals_);
  }
  template <size_t I, class = enable_if_interval<I>>
  NDARRAY_HOST_DEVICE const auto& interval() const {
    return std::get<I>(intervals_);
  }

  /** Get a specific interval of this bounds with a runtime interval index `d`.
   * This will lose knowledge of any compile-time constant interval
   * attributes, and it is not a reference to the original interval. */
  NDARRAY_HOST_DEVICE nda::interval<> interval(size_t i) const {
    assert(i < rank());
    return internal::tuple_to_array<nda::interval<>>(intervals_)[i];
  }

  /** Get a tuple of all of the intervals of this bounds. */
  NDARRAY_HOST_DEVICE intervals_type& intervals() { return intervals_; }
  NDARRAY_HOST_DEVICE const intervals_type& intervals() const { return intervals_; }

  /** Makes a tuple of dims out of each interval in this bounds. */
  NDARRAY_HOST_DEVICE auto dims() const { return internal::dims(intervals(), interval_indices()); }

  NDARRAY_HOST_DEVICE index_type min() const {
    return internal::mins(intervals(), interval_indices());
  }
  NDARRAY_HOST_DEVICE index_type max() const {
    return internal::maxs(intervals(), interval_indices());
  }
  NDARRAY_HOST_DEVICE index_type extent() const {
    return internal::extents(intervals(), interval_indices());
  }

  // TODO(jiawen):
  // Add convenience functions: width(), height(), depth.
};

// TODO(jiawen):
// operator==, !=

/** An arbitrary `bounds` with the specified rank `Rank`. This bounds is
 * compatible with any other bounds of the same rank. */
template <size_t Rank>
using bounds_of_rank = decltype(make_bounds_from_tuple(internal::tuple_of_n<interval<>, Rank>()));

/** Helper function to make a shape from a bounds. */
template <class... Intervals>
NDARRAY_HOST_DEVICE auto make_shape(const bounds<Intervals...>& bounds) {
  return make_shape_from_tuple(bounds.dims());
}

// TODO(jiawen): Plumb bounds into:
// - array
//   - make_array
// - array_ref
//   - make_array_ref

// TODO(jiawen): Implement some geometry tools:
// - standardized(bounds) -> bounds with all positive extents
// - volume (just size)

// Move into geometry.h?

using rect = bounds_of_rank<2>;
using box = bounds_of_rank<3>;
template <size_t Rank>
using hyper_rect = bounds_of_rank<Rank>;

} // namespace nda

#endif // NDARRAY_BOUNDS_H

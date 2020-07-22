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

/** \file array.h
 * \brief Main header for array library
*/

#ifndef NDARRAY_ARRAY_H
#define NDARRAY_ARRAY_H

#include <array>
#include <cassert>
#include <limits>
#include <memory>
#include <tuple>

// If we have __has_feature, automatically disable exceptions.
#ifdef __has_feature
#if !__has_feature(cxx_exceptions)
#ifndef NDARRAY_NO_EXCEPTIONS
#define NDARRAY_NO_EXCEPTIONS
#endif
#endif
#endif

#ifdef NDARRAY_NO_EXCEPTIONS
#define NDARRAY_THROW_OUT_OF_RANGE(m) do { assert(!m); abort(); } while(0)
#define NDARRAY_THROW_BAD_ALLOC() do { assert(!"bad alloc"); abort(); } while(0)
#else
#define NDARRAY_THROW_OUT_OF_RANGE(m) throw std::out_of_range(m)
#define NDARRAY_THROW_BAD_ALLOC() throw std::bad_alloc();
#endif

// Some things in this header are unbearably slow without optimization if they
// don't get inlined.
#if defined(__GNUC__) || defined(__clang__)
#define NDARRAY_INLINE inline __attribute__((always_inline))
#else
#define NDARRAY_INLINE inline
#endif

namespace nda {

using size_t = std::size_t;
/** When `NDARRAY_INT_INDICES` is defined, array indices are `int` values, otherwise
 * they are `std::ptrdiff_t`. std::ptrdiff_t is helpful for the compiler to
 * optimize address arithmetic, because it has the same size as a pointer. */
#ifdef NDARRAY_INT_INDICES
using index_t = int;
#else
using index_t = std::ptrdiff_t;
#endif

/** This value indicates a compile-time constant parameter is an unknown value,
 * and to use the corresponding runtime value instead. If a compile-time constant
 * value is not `dynamic`, it is said to be `static`. A runtime value is said to be
 * 'compatible with' a compile-time constant value if the values are equal, or the
 * compile-time constant value is dynamic. */
// It would be better to use a more unreasonable value that would never be
// used in practice, or find a better way to express this pattern.
// (https://github.com/dsharlet/array/issues/9).
constexpr index_t dynamic = -9;

// Deprecated name for `dynamic`.
constexpr index_t UNK = dynamic;

namespace internal {

NDARRAY_INLINE index_t abs(index_t a) { return a >= 0 ? a : -a; }

NDARRAY_INLINE constexpr index_t is_static(index_t x) { return x != dynamic; }
NDARRAY_INLINE constexpr index_t is_dynamic(index_t x) { return x == dynamic; }

// Given a compile-time static value, reconcile a compile-time static value and
// runtime value.
template <index_t Value>
NDARRAY_INLINE constexpr index_t reconcile(index_t value) {
  // It would be nice to assert here that Value == value. But, this is used in
  // the innermost loops, so when asserts are on, this ruins performance. It
  // is also a less helpful place to catch errors, because the context of the
  // bug is lost here.
  return is_static(Value) ? Value : value;
}

constexpr bool is_dynamic(index_t a, index_t b) { return is_dynamic(a) || is_dynamic(b); }

template <index_t A, index_t B>
using enable_if_compatible = typename std::enable_if<is_dynamic(A, B) || A == B>::type;

// Math for (possibly) static values.
constexpr index_t static_add(index_t a, index_t b) { return is_dynamic(a, b) ? dynamic : a + b; }
constexpr index_t static_sub(index_t a, index_t b) { return is_dynamic(a, b) ? dynamic : a - b; }
constexpr index_t static_mul(index_t a, index_t b) { return is_dynamic(a, b) ? dynamic : a * b; }
constexpr index_t static_min(index_t a, index_t b) {
  return is_dynamic(a, b) ? dynamic : (a < b ? a : b);
}
constexpr index_t static_max(index_t a, index_t b) {
  return is_dynamic(a, b) ? dynamic : (a > b ? a : b);
}

}  // namespace internal

/** An iterator representing an index. */
class index_iterator {
  index_t i_;

 public:
  /** Construct the iterator with an index `i`. */
  index_iterator(index_t i) : i_(i) {}

  /** Access the current index of this iterator. */
  NDARRAY_INLINE index_t operator*() const { return i_; }

  NDARRAY_INLINE bool operator==(const index_iterator& r) const { return i_ == r.i_; }
  NDARRAY_INLINE bool operator!=(const index_iterator& r) const { return i_ != r.i_; }

  NDARRAY_INLINE index_iterator operator++(int) { return index_iterator(i_++); }
  NDARRAY_INLINE index_iterator& operator++() { ++i_; return *this; }
};

/** Describes a half-open interval of indices. The template parameters enable
 * providing compile time constants for the `min` and `extent` of the interval.
 * The values in the interval `[min, min + extent)` are considered in bounds.
 *
 * `interval<> a` is said to be 'compatible with' another `interval<Min, Extent> b` if
 * `a.min()` is compatible with `Min` and `a.extent()` is compatible with `Extent`.
 *
 * Examples:
 * - `interval<>` is an interval with runtime-valued `min` and `extent`.
 * - `interval<0>` is an interval with compile-time constant `min` of 0, and
 *   runtime-valued `extent`.
 * - `interval<dynamic, 8>` is an interval with compile-time constant `extent of 8
 *   and runtime-valued `min`.
 * - `interval<2, 3>` is a fully compile-time constant interval of indices 2, 3, 4. */
template <index_t Min_ = dynamic, index_t Extent_ = dynamic>
class interval {
 protected:
  index_t min_;
  index_t extent_;

 public:
  static constexpr index_t Min = Min_;
  static constexpr index_t Extent = Extent_;
  static constexpr index_t Max =
      internal::static_sub(internal::static_add(Min, Extent), 1);

  /** Construct a new interval object. If the `min` or `extent` is specified in
   * the constructor, it must have a value compatible with `Min` or `Extent`,
   * respectively.
   *
   * The default values if not specified in the constructor are:
   * - The default `min` is `Min` if `Min` is static, or 0 if not.
   * - The default `extent` is `Extent` if `Extent` is static, or 1 if not. */
  interval(index_t min, index_t extent) {
    set_min(min);
    set_extent(extent);
  }
  interval(index_t min) : interval(min, internal::is_static(Extent) ? Extent : 1) {}
  interval() : interval(internal::is_static(Min) ? Min : 0) {}

  interval(const interval&) = default;
  interval(interval&&) = default;
  interval& operator=(const interval&) = default;
  interval& operator=(interval&&) = default;

  /** Copy construction or assignment of another interval object, possibly
   * with different compile-time template parameters. `other.min()` and
   * `other.extent()` must be compatible with `Min` and `Extent`, respectively. */
  template <index_t CopyMin, index_t CopyExtent,
      class = internal::enable_if_compatible<Min, CopyMin>,
      class = internal::enable_if_compatible<Extent, CopyExtent>>
  interval(const interval<CopyMin, CopyExtent>& other) : interval(other.min(), other.extent()) {}
  template <index_t CopyMin, index_t CopyExtent,
      class = internal::enable_if_compatible<Min, CopyMin>,
      class = internal::enable_if_compatible<Extent, CopyExtent>>
  interval& operator=(const interval<CopyMin, CopyExtent>& other) {
    set_min(other.min());
    set_extent(other.extent());
    return *this;
  }

  /** Get or set the first index in this interval. */
  NDARRAY_INLINE index_t min() const { return internal::reconcile<Min>(min_); }
  NDARRAY_INLINE void set_min(index_t min) {
    if (internal::is_dynamic(Min)) {
      min_ = min;
    } else {
      assert(min == Min);
    }
  }
  /** Get or set the number of indices in this interval. */
  NDARRAY_INLINE index_t extent() const { return internal::reconcile<Extent>(extent_); }
  NDARRAY_INLINE void set_extent(index_t extent) {
    if (internal::is_dynamic(Extent)) {
      extent_ = extent;
    } else {
      assert(extent == Extent);
    }
  }
  /** Get or set the last index in this interval. */
  NDARRAY_INLINE index_t max() const { return min() + extent() - 1; }
  NDARRAY_INLINE void set_max(index_t max) { set_extent(max - min() + 1); }

  /** Returns true if `at` is within the interval `[min(), max()]`. */
  NDARRAY_INLINE bool is_in_range(index_t at) const { return min() <= at && at <= max(); }
  /** Returns true if `at.min()` and `at.max()` are both within the interval
   * `[min(), max()]`. */
  template <index_t OtherMin, index_t OtherExtent>
  NDARRAY_INLINE bool is_in_range(const interval<OtherMin, OtherExtent>& at) const {
    return min() <= at.min() && at.max() <= max();
  }

  /** Make an iterator referring to the first index in this interval. */
  index_iterator begin() const { return index_iterator(min()); }
  /** Make an iterator referring to one past the last index in this interval. */
  index_iterator end() const { return index_iterator(max() + 1); }

  /** Two interval objects are considered equal if they contain the
   * same indices. */
  template <index_t OtherMin, index_t OtherExtent,
      class = internal::enable_if_compatible<Min, OtherMin>,
      class = internal::enable_if_compatible<Extent, OtherExtent>>
  bool operator==(const interval<OtherMin, OtherExtent>& other) const {
    return min() == other.min() && extent() == other.extent();
  }
  template <index_t OtherMin, index_t OtherExtent,
      class = internal::enable_if_compatible<Min, OtherMin>,
      class = internal::enable_if_compatible<Extent, OtherExtent>>
  bool operator!=(const interval<OtherMin, OtherExtent>& other) const {
    return !operator==(other);
  }
};

/** An alias of `interval` with a fixed extent and dynamic min. */
template <index_t Extent>
using fixed_interval = interval<dynamic, Extent>;

/** Make an interval from a half-open range `[begin, end)`. */
inline interval<> range(index_t begin, index_t end) {
  return interval<>(begin, end - begin);
}
inline interval<> r(index_t begin, index_t end) {
  return interval<>(begin, end - begin);
}

/** Make an interval from a half-open range `[begin, begin + Extent)`. */
template <index_t Extent>
fixed_interval<Extent> range(index_t begin) {
  return fixed_interval<Extent>(begin);
}
template <index_t Extent>
fixed_interval<Extent> r(index_t begin) {
  return fixed_interval<Extent>(begin);
}

/** Placeholder object representing an interval that indicates keep
 * the whole dimension when used in an indexing expression. */
const interval<0, -1> all, _;

/** Overloads of `std::begin` and `std::end` for an interval. */
template <index_t Min, index_t Extent>
index_iterator begin(const interval<Min, Extent>& d) { return d.begin(); }
template <index_t Min, index_t Extent>
index_iterator end(const interval<Min, Extent>& d) { return d.end(); }

/** Clamp `x` to the interval [min, max]. */
inline index_t clamp(index_t x, index_t min, index_t max) {
  return std::min(std::max(x, min), max);
}

/** Clamp `x` to the range described by an object `r` with a `min()` and
 * `max()` method. */
template <class Range>
index_t clamp(index_t x, const Range& r) {
  return clamp(x, r.min(), r.max());
}

namespace internal {

// An iterator for a range of intervals.
template <index_t InnerExtent = dynamic>
class split_iterator {
  fixed_interval<InnerExtent> i;
  index_t outer_max;

 public:
  split_iterator(const fixed_interval<InnerExtent>& i, index_t outer_max)
      : i(i), outer_max(outer_max) {}

  bool operator==(const split_iterator& r) const { return i.min() == r.i.min(); }
  bool operator!=(const split_iterator& r) const { return i.min() != r.i.min(); }

  fixed_interval<InnerExtent> operator *() const { return i; }

  split_iterator& operator++() {
    if (is_static(InnerExtent)) {
      // When the extent of the inner split is a compile-time constant,
      // we can't shrink the out of bounds interval. Instead, shift the min,
      // assuming the outer dimension is bigger than the inner extent.
      i.set_min(i.min() + InnerExtent);
      // Only shift the min when this straddles the end of the buffer,
      // so the iterator can advance to the end (one past the max).
      if (i.min() <= outer_max && i.max() > outer_max) {
        i.set_min(outer_max - InnerExtent + 1);
      }
    } else {
      // When the extent of the inner split is not a compile-time constant,
      // we can just modify the extent.
      i.set_min(i.min() + i.extent());
      index_t max = std::min(i.max(), outer_max);
      i.set_extent(max - i.min() + 1);
    }
    return *this;
  }
  split_iterator operator++(int) {
    split_iterator<InnerExtent> result(i);
    ++result;
    return result;
  }
};

// TODO: Remove this when std::iterator_range is standard.
template <class T>
class iterator_range {
  T begin_;
  T end_;

 public:
  iterator_range(T begin, T end) : begin_(begin), end_(end) {}

  T begin() const { return begin_; }
  T end() const { return end_; }
};

template <index_t InnerExtent = dynamic>
using split_iterator_range = iterator_range<split_iterator<InnerExtent>>;

}  // namespace internal

/** Split an interval `v` into an iteratable range of intervals by a compile-time
 * constant `InnerExtent`. If `InnerExtent` does not divide `v.extent()`,
 * the last interval will be shifted to overlap with the second-to-last iteration,
 * to preserve the compile-time constant extent, which implies `v.extent()`
 * must be larger `InnerExtent`.
 *
 * Examples:
 * - `split<4>(interval<>(0, 8))` produces the intervals `[0, 4)`, `[4, 8)`.
 * - `split<5>(interval<>(0, 12))` produces the intervals `[0, 5)`,
 *   `[5, 10)`, `[7, 12)`. Note the last two intervals overlap. */
template <index_t InnerExtent, index_t Min, index_t Extent>
internal::split_iterator_range<InnerExtent> split(const interval<Min, Extent>& v) {
  assert(v.extent() >= InnerExtent);
  return {
      {fixed_interval<InnerExtent>(v.min()), v.max()},
      {fixed_interval<InnerExtent>(v.max() + 1), v.max()}};
}

/** Split an interval `v` into an iterable range of intervals by `inner_extent`. If
 * `inner_extent` does not divide `v.extent()`, the last iteration will be
 * clamped to the outer interval.
 *
 * Examples:
 * - `split(interval<>(0, 12), 5)` produces the intervals `[0, 5)`,
 * `  [5, 10)`, `[10, 12)`. */
// TODO: This probably doesn't need to be templated, but it might help
// avoid some conversion messes. dim<Min, Extent> probably can't implicitly
// convert to interval<>.
template <index_t Min, index_t Extent>
internal::split_iterator_range<> split(const interval<Min, Extent>& v, index_t inner_extent) {
  return {
      {interval<>(v.min(), inner_extent), v.max()},
      {interval<>(v.max() + 1, inner_extent), v.max()}};
}

/** Describes one dimension of an array. The template parameters enable
 * providing compile time constants for the `min`, `extent`, and `stride` of the
 * dim.
 *
 * These parameters define a mapping from the indices of the dimension to
 * offsets: `offset(x) = (x - min)*stride`. The extent does not affect the
 * mapping directly. Values not in the interval `[min, min + extent)` are considered
 * to be out of bounds. */
// TODO: Consider adding helper class constant<Value> to use for the members of
// dim. (https://github.com/dsharlet/array/issues/1)
template <index_t Min_ = dynamic, index_t Extent_ = dynamic, index_t Stride_ = dynamic>
class dim : public interval<Min_, Extent_> {
 protected:
  index_t stride_;

 public:
  using base_range = interval<Min_, Extent_>;

  using base_range::Min;
  using base_range::Extent;
  using base_range::Max;

  static constexpr index_t Stride = Stride_;

  /** Construct a new dim object. If the `min`, `extent` or `stride` are
   * specified in the constructor, it must have a value compatible with `Min`,
   * `Extent`, or `Stride`, respectively.
   *
   * The default values if not specified in the constructor are:
   * - The default `min` is `Min` if `Min` is static, or 0 if not.
   * - The default `extent` is `Extent` if `Extent` is static, or 0 if not.
   * - The default `stride` is `Stride`. */
  dim(index_t min, index_t extent, index_t stride = Stride) : base_range(min, extent) {
    set_stride(stride);
  }
  dim(index_t extent) : dim(internal::is_static(Min) ? Min : 0, extent) {}
  dim() : dim(internal::is_static(Extent) ? Extent : 0) {}

  dim(const base_range& interval, index_t stride = Stride)
      : dim(interval.min(), interval.extent(), stride) {}
  dim(const dim&) = default;
  dim(dim&&) = default;
  dim& operator=(const dim&) = default;
  dim& operator=(dim&&) = default;

  /** Copy construction or assignment of another dim object, possibly
   * with different compile-time template parameters. `other.min()`,
   * `other.extent()`, and `other.stride()` must be compatible with `Min`,
   * `Extent`, and `Stride`, respectively. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride,
      class = internal::enable_if_compatible<Min, CopyMin>,
      class = internal::enable_if_compatible<Extent, CopyExtent>,
      class = internal::enable_if_compatible<Stride, CopyStride>>
  dim(const dim<CopyMin, CopyExtent, CopyStride>& other)
      : dim(other.min(), other.extent(), other.stride()) {}
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride,
      class = internal::enable_if_compatible<Min, CopyMin>,
      class = internal::enable_if_compatible<Extent, CopyExtent>,
      class = internal::enable_if_compatible<Stride, CopyStride>>
  dim& operator=(const dim<CopyMin, CopyExtent, CopyStride>& other) {
    set_min(other.min());
    set_extent(other.extent());
    set_stride(other.stride());
    return *this;
  }

  using base_range::set_min;
  using base_range::set_extent;
  using base_range::set_max;
  using base_range::min;
  using base_range::max;
  using base_range::extent;
  using base_range::begin;
  using base_range::end;
  using base_range::is_in_range;

  /** Get or set the distance in flat indices between neighboring elements
   * in this dim. */
  NDARRAY_INLINE index_t stride() const { return internal::reconcile<Stride>(stride_); }
  NDARRAY_INLINE void set_stride(index_t stride) {
    if (internal::is_dynamic(Stride)) {
      stride_ = stride;
    } else {
      assert(stride == Stride);
    }
  }

  /** Offset of the index `at` in this dim in the flat array. */
  NDARRAY_INLINE index_t flat_offset(index_t at) const { return (at - min()) * stride(); }

  /** Two dim objects are considered equal if their mins, extents, and strides
   * are equal. */
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride,
      class = internal::enable_if_compatible<Min, OtherMin>,
      class = internal::enable_if_compatible<Extent, OtherExtent>,
      class = internal::enable_if_compatible<Stride, OtherStride>>
  bool operator==(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
    return min() == other.min() && extent() == other.extent() && stride() == other.stride();
  }
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride,
      class = internal::enable_if_compatible<Min, OtherMin>,
      class = internal::enable_if_compatible<Extent, OtherExtent>,
      class = internal::enable_if_compatible<Stride, OtherStride>>
  bool operator!=(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
    return !operator==(other);
  }
};

/** Alias of `dim` where the min is not specified at compile time. */
template <index_t Extent, index_t Stride = dynamic>
using fixed_dim = dim<dynamic, Extent, Stride>;

/** Alias of `dim` where the compile-time stride parameter is known
 * to be one. */
template <index_t Min = dynamic, index_t Extent = dynamic>
using dense_dim = dim<Min, Extent, 1>;

/** Alias of `dim` where only the stride parameter is specified at
 * compile time. */
template <index_t Stride>
using strided_dim = dim<dynamic, dynamic, Stride>;

/** Alias of `dim` where the compile-time stride parameter is known
 * to be zero. */
template <index_t Min = dynamic, index_t Extent = dynamic>
using broadcast_dim = dim<Min, Extent, 0>;

namespace internal {

using std::index_sequence;
using std::make_index_sequence;

// Call `fn` with the elements of tuple `args` unwrapped from the tuple.
// TODO: When we assume C++17, this can be replaced by std::apply.
template <class Fn, class Args, size_t... Is>
NDARRAY_INLINE auto apply(Fn&& fn, const Args& args, index_sequence<Is...>)
    -> decltype(fn(std::get<Is>(args)...)) {
  return fn(std::get<Is>(args)...);
}
template <class Fn, class... Args>
NDARRAY_INLINE auto apply(Fn&& fn, const std::tuple<Args...>& args)
    -> decltype(apply(fn, args, make_index_sequence<sizeof...(Args)>())) {
  return apply(fn, args, make_index_sequence<sizeof...(Args)>());
}

template <class Fn, class... Args>
using enable_if_callable = decltype(std::declval<Fn>()(std::declval<Args>()...));
template <class Fn, class Args>
using enable_if_applicable = decltype(apply(std::declval<Fn>(), std::declval<Args>()));

// Some variadic reduction helpers.
NDARRAY_INLINE constexpr index_t sum() { return 0; }
template <class... Rest>
NDARRAY_INLINE constexpr index_t sum(index_t first, Rest... rest) {
  return first + sum(rest...);
}

NDARRAY_INLINE constexpr index_t product() { return 1; }
template <class... Rest>
NDARRAY_INLINE constexpr index_t product(index_t first, Rest... rest) {
  return first * product(rest...);
}

NDARRAY_INLINE constexpr index_t variadic_min() { return std::numeric_limits<index_t>::max(); }
template <class... Rest>
NDARRAY_INLINE constexpr index_t variadic_min(index_t first, Rest... rest) {
  return std::min(first, variadic_min(rest...));
}

NDARRAY_INLINE constexpr index_t variadic_max() { return std::numeric_limits<index_t>::min(); }
template <class... Rest>
NDARRAY_INLINE constexpr index_t variadic_max(index_t first, Rest... rest) {
  return std::max(first, variadic_max(rest...));
}

// Computes the product of the extents of the dims.
template <class Tuple, size_t... Is>
index_t product(const Tuple& t, index_sequence<Is...>) {
  return product(std::get<Is>(t)...);
}

// Returns true if all of bools are true.
template <class... Bools>
bool all(Bools... bools) {
  return sum((bools ? 0 : 1)...) == 0;
}

// Computes the sum of the offsets of a list of dims and indices.
template <class Dims, class Indices, size_t... Is>
index_t flat_offset_tuple(const Dims& dims, const Indices& indices, index_sequence<Is...>) {
  return sum(std::get<Is>(dims).flat_offset(std::get<Is>(indices))...);
}

template <size_t D, class Dims>
NDARRAY_INLINE index_t flat_offset_pack(const Dims& dims) {
  return 0;
}
template <size_t D, class Dims, class... Indices>
NDARRAY_INLINE index_t flat_offset_pack(const Dims& dims, index_t i0, Indices... indices) {
  return std::get<D>(dims).flat_offset(i0) + flat_offset_pack<D + 1>(dims, indices...);
}

// Computes one more than the sum of the offsets of the last index in every dim.
template <class Dims, size_t... Is>
index_t flat_min(const Dims& dims, index_sequence<Is...>) {
  return sum(
      (std::get<Is>(dims).extent() - 1) *
      std::min(static_cast<index_t>(0), std::get<Is>(dims).stride())...);
}

template <class Dims, size_t... Is>
index_t flat_max(const Dims& dims, index_sequence<Is...>) {
  return sum(
      (std::get<Is>(dims).extent() - 1) *
      std::max(static_cast<index_t>(0), std::get<Is>(dims).stride())...);
}

// Make dims with the interval of the first parameter and the stride
// of the second parameter.
template <index_t DimMin, index_t DimExtent, index_t DimStride>
auto range_with_stride(index_t x, const dim<DimMin, DimExtent, DimStride>& d) {
  return dim<dynamic, 1, DimStride>(x, 1, d.stride());
}
template <index_t CropMin, index_t CropExtent, index_t DimMin, index_t DimExtent, index_t Stride>
auto range_with_stride(
    const interval<CropMin, CropExtent>& x, const dim<DimMin, DimExtent, Stride>& d) {
  return dim<CropMin, CropExtent, Stride>(x.min(), x.extent(), d.stride());
}
template <index_t Min, index_t Extent, index_t Stride>
auto range_with_stride(const decltype(_)&, const dim<Min, Extent, Stride>& d) {
  return d;
}

template <class Intervals, class Dims, size_t... Is>
auto intervals_with_strides(const Intervals& intervals, const Dims& dims, index_sequence<Is...>) {
  return std::make_tuple(range_with_stride(std::get<Is>(intervals), std::get<Is>(dims))...);
}

// Make a tuple of dims corresponding to elements in intervals that are not slices.
template <class Dim>
std::tuple<> skip_slices_impl(const Dim& dim, index_t) { return std::tuple<>(); }
template <class Dim>
std::tuple<Dim> skip_slices_impl(const Dim& dim, const interval<>&) { return std::tuple<Dim>(dim); }

template <class Dims, class Intervals, size_t... Is>
auto skip_slices(const Dims& dims, const Intervals& intervals, index_sequence<Is...>) {
  return std::tuple_cat(skip_slices_impl(std::get<Is>(dims), std::get<Is>(intervals))...);
}

// Checks if all indices are in interval of each corresponding dim.
template <class Dims, class Indices, size_t... Is>
bool is_in_range(const Dims& dims, const Indices& indices, index_sequence<Is...>) {
  return all(std::get<Is>(dims).is_in_range(std::get<Is>(indices))...);
}

// Get the mins of a series of intervals.
template <class Dim>
index_t min_of_range(index_t x, const Dim&) { return x; }
template <index_t Min, index_t Extent, class Dim>
index_t min_of_range(const interval<Min, Extent>& x, const Dim&) { return x.min(); }
template <class Dim>
index_t min_of_range(const decltype(_)&, const Dim& dim) { return dim.min(); }

template <class Intervals, class Dims, size_t... Is>
auto mins_of_intervals(const Intervals& intervals, const Dims& dims, index_sequence<Is...>) {
  return std::make_tuple(min_of_range(std::get<Is>(intervals), std::get<Is>(dims))...);
}

template <class... Dims, size_t... Is>
auto mins(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).min()...);
}

template <class... Dims, size_t... Is>
auto extents(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).extent()...);
}

template <class... Dims, size_t... Is>
auto strides(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).stride()...);
}

template <class... Dims, size_t... Is>
auto maxs(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).max()...);
}

// The following series of functions implements the algorithm for
// automatically determining what unknown dynamic strides should be.

// A proposed stride is "OK" w.r.t. `dim` if the proposed
// stride does not intersect the dim.
template <class Dim>
bool is_stride_ok(index_t stride, index_t extent, const Dim& dim) {
  if (is_dynamic(dim.stride())) {
    // If the dimension has an unknown dynamic stride, it's OK, we're
    // resolving the current dim first.
    return true;
  }
  if (dim.extent() * abs(dim.stride()) <= stride) {
    // The dim is completely inside the proposed stride.
    return true;
  }
  index_t flat_extent = extent * stride;
  if (abs(dim.stride()) >= flat_extent) {
    // The dim is completely outside the proposed stride.
    return true;
  }
  return false;
}

template <class AllDims, size_t... Is>
bool is_stride_ok(index_t stride, index_t extent, const AllDims& all_dims, index_sequence<Is...>) {
  return all(is_stride_ok(stride, extent, std::get<Is>(all_dims))...);
}

// Replace strides that are not OK with values that cannot be the
// smallest stride.
template <class AllDims>
index_t filter_stride(index_t stride, index_t extent, const AllDims& all_dims) {
  constexpr size_t rank = std::tuple_size<AllDims>::value;
  if (is_stride_ok(stride, extent, all_dims, make_index_sequence<rank>())) {
    return stride;
  } else {
    return std::numeric_limits<index_t>::max();
  }
}

// The candidate stride for some other dimension is the minimum stride it
// could have without intersecting this dim.
template <class Dim>
index_t candidate_stride(const Dim& dim) {
  if (is_dynamic(dim.stride())) {
    return std::numeric_limits<index_t>::max();
  }
  return std::max(static_cast<index_t>(1), abs(dim.stride()) * dim.extent());
}

// Find the best stride (the smallest) out of all possible candidate strides.
template <class AllDims, size_t... Is>
index_t find_stride(index_t extent, const AllDims& all_dims, index_sequence<Is...>) {
  return variadic_min(
      filter_stride(1, extent, all_dims),
      filter_stride(candidate_stride(std::get<Is>(all_dims)), extent, all_dims)...);
}

// Replace unknown dynamic strides for each dimension, starting with the first dimension.
template <class AllDims>
void resolve_unknown_strides(AllDims& all_dims) {}
template <class AllDims, class Dim0, class... Dims>
void resolve_unknown_strides(AllDims& all_dims, Dim0& dim0, Dims&... dims) {
  if (is_dynamic(dim0.stride())) {
    constexpr size_t rank = std::tuple_size<AllDims>::value;
    dim0.set_stride(find_stride(dim0.extent(), all_dims, make_index_sequence<rank>()));
  }
  resolve_unknown_strides(all_dims, dims...);
}

template <class Dims, size_t... Is>
void resolve_unknown_strides(Dims& dims, index_sequence<Is...>) {
  resolve_unknown_strides(dims, std::get<Is>(dims)...);
}

template<class Dims, size_t... Is>
bool is_resolved(const Dims& dims, index_sequence<Is...>) {
  return all(!is_dynamic(std::get<Is>(dims).stride())...);
}

// A helper to transform an array to a tuple.
template <class T, class Tuple, size_t... Is>
std::array<T, sizeof...(Is)> tuple_to_array(const Tuple& t, index_sequence<Is...>) {
  return {{std::get<Is>(t)...}};
}

template <class T, class... Ts>
std::array<T, sizeof...(Ts)> tuple_to_array(const std::tuple<Ts...>& t) {
  return tuple_to_array<T>(t, make_index_sequence<sizeof...(Ts)>());
}

template <class T, size_t N, size_t... Is>
auto array_to_tuple(const std::array<T, N>& a, index_sequence<Is...>) {
  return std::make_tuple(a[Is]...);
}

template <class T, size_t N>
auto array_to_tuple(const std::array<T, N>& a) {
  return array_to_tuple(a, make_index_sequence<N>());
}

template<class T, size_t N>
struct tuple_of_n {
  using rest = typename tuple_of_n<T, N - 1>::type;
  using type = decltype(std::tuple_cat(std::declval<std::tuple<T>>(), std::declval<rest>()));
};

template<class T>
struct tuple_of_n<T, 0> {
  using type = std::tuple<>;
};

// A helper to check if a parameter pack is entirely implicitly convertible to type T, for use with
// std::enable_if
template <class T, class... Args>
struct all_of_type : std::false_type {};
template <class T>
struct all_of_type<T> : std::true_type {};
template <class T, class Arg, class... Args>
struct all_of_type<T, Arg, Args...> {
  static constexpr bool value =
      std::is_constructible<T, Arg>::value && all_of_type<T, Args...>::value;
};

template <size_t I, class T, class... Us, std::enable_if_t<(I < sizeof...(Us)), int> = 0>
auto convert_dim(const std::tuple<Us...>& u) {
  return std::get<I>(u);
}
template <size_t I, class T, class... Us, std::enable_if_t<(I >= sizeof...(Us)), int> = 0>
auto convert_dim(const std::tuple<Us...>& u) {
  // For dims beyond the rank of U, make a dimension of type T_I with extent 1.
  return decltype(std::get<I>(std::declval<T>()))(1);
}

template <class T, class U, size_t... Is>
T convert_dims(const U& u, std::index_sequence<Is...>) {
  return std::make_tuple(convert_dim<Is, T>(u)...);
}

constexpr index_t factorial(index_t x) {
  return x == 1 ? 1 : x * factorial(x - 1);
}

// The errors that result from not satisfying this check are probably hell,
// but it would be pretty tricky to check that all of [0, Rank) is in `Is...`
template <size_t Rank, size_t... Is>
using enable_if_permutation = typename std::enable_if<
    sizeof...(Is) == Rank && product((Is + 2)...) == factorial(Rank + 1)>::type;

}  // namespace internal

template <class... Dims>
class shape;

/** Helper function to make a tuple from a variadic list of `dims...`. */
template <class... Dims>
auto make_shape(Dims... dims) {
  return shape<Dims...>(dims...);
}

template <class... Dims>
shape<Dims...> make_shape_from_tuple(const std::tuple<Dims...>& dims) {
  return shape<Dims...>(dims);
}

/** Type of an index for an array of rank `Rank`. This will be
 * `std::tuple<...>` with `Rank` `index_t` values.
 *
 * For example, `index_of_rank<3>` is `std::tuple<index_t, index_t, index_t>`. */
template <size_t Rank>
using index_of_rank = typename internal::tuple_of_n<index_t, Rank>::type;

/** A list of `Dim` objects describing a multi-dimensional space of indices.
 * The `rank` of a shape refers to the number of dimensions in the shape.
 * The first dimension is known as the 'innermost' dimension, and dimensions
 * then increase until the 'outermost' dimension.
 *
 * Shapes map a multi-dimensional index `x` to a flat offset by
 * `sum(dim<i>().flat_offset(std::get<i>(x)))` for `i in [0, Rank)`. */
template <class... Dims>
class shape {
 public:
  /** The type of the dims tuple of this shape. */
  using dims_type = std::tuple<Dims...>;

  /** Number of dims in this shape. */
  static constexpr size_t rank() { return std::tuple_size<dims_type>::value; }

  /** A shape is scalar if its rank is 0. */
  static constexpr bool is_scalar() { return rank() == 0; }

  /** The type of an index for this shape. */
  using index_type = index_of_rank<rank()>;

  using size_type = size_t;

 private:
  dims_type dims_;

  // TODO: This should use std::is_constructible<dims_type, std::tuple<OtherDims...>>
  // but it is broken on some compilers (https://github.com/dsharlet/array/issues/20).
  template <class... OtherDims>
  using enable_if_dims_compatible = typename std::enable_if<sizeof...(OtherDims) == rank()>::type;

  template <class... Args>
  using enable_if_same_rank = typename std::enable_if<(sizeof...(Args) == rank())>::type;

  template <class... Args>
  using enable_if_indices =
      typename std::enable_if<internal::all_of_type<index_t, Args...>::value>::type;

  template <class... Args>
  using enable_if_slices = typename std::enable_if<
      internal::all_of_type<interval<>, Args...>::value &&
      !internal::all_of_type<index_t, Args...>::value>::type;

  template <size_t Dim>
  using enable_if_dim = typename std::enable_if<(Dim < rank())>::type;

 public:
  shape() {}
  // TODO: This is a bit messy, but necessary to avoid ambiguous default
  // constructors when Dims is empty.
  template <size_t N = sizeof...(Dims), class = typename std::enable_if<(N > 0)>::type>
  shape(const Dims&... dims) : dims_(dims...) {}
  shape(const shape&) = default;
  shape(shape&&) = default;
  shape& operator=(const shape&) = default;
  shape& operator=(shape&&) = default;

  /** Construct or assign a shape from another set of dims of a possibly
   * different type. Each dim must be compatible with the corresponding
   * dim of this shape. */
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape(const std::tuple<OtherDims...>& other) : dims_(other) {}
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape(OtherDims... other_dims) : dims_(other_dims...) {}
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape(const shape<OtherDims...>& other) : dims_(other.dims()) {}
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape& operator=(const shape<OtherDims...>& other) {
    dims_ = other.dims();
    return *this;
  }

  // We cannot have an dims_type constructor because it will be
  // ambiguous with the Dims... constructor for 1D shapes.

  /** Replace strides with automatically determined values.
   *
   * An automatic stride for a dimension is determined by taking the minimum
   * of all possible candidate strides, which are the product of the stride
   * and extent of all dimensions with a known stride. This is repeated for
   * each dimension, starting with the innermost dimension.
   *
   * Examples:
   * - `{{0, 5}, {0, 10}}` -> `{{0, 5, 1}, {0, 10, 5}}`
   * - `{{0, 5}, {0, 10}, {0, 3, 1}}` -> `{{0, 5, 3}, {0, 10, 15}, {0, 3, 1}}` */
  void resolve() {
    internal::resolve_unknown_strides(dims_, internal::make_index_sequence<rank()>());
  }

  /** Check if all strides of the shape are known. */
  bool is_resolved() const {
    return internal::is_resolved(dims_, internal::make_index_sequence<rank()>());
  }

  /** Returns `true` if the indices or intervals `args` are in interval of this shape. */
  template <class... Args, class = enable_if_same_rank<Args...>>
  bool is_in_range(const std::tuple<Args...>& args) const {
    return internal::is_in_range(dims_, args, internal::make_index_sequence<rank()>());
  }
  template <class... Args, class = enable_if_same_rank<Args...>>
  bool is_in_range(Args... args) const {
    return internal::is_in_range(
        dims_, std::make_tuple(args...), internal::make_index_sequence<rank()>());
  }

  /** Compute the flat offset of the index `indices`. */
  index_t operator() (const index_type& indices) const {
    return internal::flat_offset_tuple(dims_, indices, internal::make_index_sequence<rank()>());
  }
  index_t operator[] (const index_type& indices) const {
    return internal::flat_offset_tuple(dims_, indices, internal::make_index_sequence<rank()>());
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  index_t operator() (Args... indices) const {
    return internal::flat_offset_pack<0>(dims_, indices...);
  }

  /** Create a new shape from this shape using a indices or intervals `args`.
   * Dimensions corresponding to indices in `args` are sliced, i.e. the result
   * will not have this dimension. The rest of the dimensions are cropped. */
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (const std::tuple<Args...>& args) const {
    auto new_dims =
        internal::intervals_with_strides(args, dims_, internal::make_index_sequence<rank()>());
    auto new_dims_no_slices =
        internal::skip_slices(new_dims, args, internal::make_index_sequence<rank()>());
    return make_shape_from_tuple(new_dims_no_slices);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator[] (const std::tuple<Args...>& args) const { return operator()(args); }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (Args... args) const { return operator()(std::make_tuple(args...)); }

  /** Get a specific dim `D` of this shape. */
  template <size_t D, class = enable_if_dim<D>>
  auto& dim() { return std::get<D>(dims_); }
  template <size_t D, class = enable_if_dim<D>>
  const auto& dim() const { return std::get<D>(dims_); }

  /** Get a specific dim of this shape with a runtime dimension index `d`.
   * This will lose knowledge of any compile-time constant dimension
   * attributes. */
  nda::dim<> dim(size_t d) const {
    assert(d < rank());
    return internal::tuple_to_array<nda::dim<>>(dims_)[d];
  }

  /** Get a tuple of all of the dims of this shape. */
  dims_type& dims() { return dims_; }
  const dims_type& dims() const { return dims_; }

  index_type min() const { return internal::mins(dims(), internal::make_index_sequence<rank()>()); }
  index_type max() const { return internal::maxs(dims(), internal::make_index_sequence<rank()>()); }
  index_type extent() const {
    return internal::extents(dims(), internal::make_index_sequence<rank()>());
  }
  index_type stride() const {
    return internal::strides(dims(), internal::make_index_sequence<rank()>());
  }

  /** Compute the min, max, or extent of the flat offsets of this shape.
   * This is the extent of the valid interval of values returned by `operator()`
   * or `operator[]`. */
  index_t flat_min() const {
    return internal::flat_min(dims_, internal::make_index_sequence<rank()>());
  }
  index_t flat_max() const {
    return internal::flat_max(dims_, internal::make_index_sequence<rank()>());
  }
  size_type flat_extent() const {
    index_t e = flat_max() - flat_min() + 1;
    return e < 0 ? 0 : static_cast<size_type>(e);
  }

  /** Compute the total number of indices in this shape. */
  size_type size() const {
    index_t s = internal::product(extent(), internal::make_index_sequence<rank()>());
    return s < 0 ? 0 : static_cast<size_type>(s);
  }

  /** A shape is empty if its size is 0. */
  bool empty() const { return size() == 0; }

  /** Returns `true` if this shape is 'compact' in memory. A shape is compact
   * if there are no unaddressable flat indices between the first and last
   * addressable flat elements. */
  bool is_compact() const { return flat_extent() <= size(); }

  /** Returns `true` if this shape is an injective function mapping indices to
   * flat indices. If the dims overlap, or a dim has stride zero, multiple
   * indices will map to the same flat index; in this case, this function will
   * return `false`. */
  bool is_one_to_one() const {
    // TODO: https://github.com/dsharlet/array/issues/2
    return flat_extent() >= size();
  }

  /** Returns `true` if this shape projects to a set of flat indices that is a
   * subset of the other shape's projection to flat indices, with an offset
   * `offset`. */
  template <typename OtherShape>
  bool is_subset_of(const OtherShape& other, index_t offset) const {
    // TODO: https://github.com/dsharlet/array/issues/2
    return flat_min() >= other.flat_min() + offset && flat_max() <= other.flat_max() + offset;
  }

  /** Provide some aliases for common interpretations of dimensions
   * `i`, `j`, `k` as dimensions 0, 1, 2, respectively. */
  auto& i() { return dim<0>(); }
  const auto& i() const { return dim<0>(); }
  auto& j() { return dim<1>(); }
  const auto& j() const { return dim<1>(); }
  auto& k() { return dim<2>(); }
  const auto& k() const { return dim<2>(); }

  /** Provide some aliases for common interpretations of dimensions
   * `x`, `y`, `z` or `c`, `w` as dimensions 0, 1, 2, 3 respectively. */
  auto& x() { return dim<0>(); }
  const auto& x() const { return dim<0>(); }
  auto& y() { return dim<1>(); }
  const auto& y() const { return dim<1>(); }
  auto& z() { return dim<2>(); }
  const auto& z() const { return dim<2>(); }
  auto& c() { return dim<2>(); }
  const auto& c() const { return dim<2>(); }
  auto& w() { return dim<3>(); }
  const auto& w() const { return dim<3>(); }

  /** Assuming this array represents an image with dimensions {width,
   * height, channels}, get the extent of those dimensions. */
  index_t width() const { return x().extent(); }
  index_t height() const { return y().extent(); }
  index_t channels() const { return c().extent(); }

  /** Assuming this array represents a matrix with dimensions {rows,
   * cols}, get the extent of those dimensions. */
  index_t rows() const { return i().extent(); }
  index_t columns() const { return j().extent(); }

  /** A shape is equal to another shape if the dim objects of each
   * dimension from both shapes are equal. */
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  bool operator==(const shape<OtherDims...>& other) const { return dims_ == other.dims(); }
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  bool operator!=(const shape<OtherDims...>& other) const { return dims_ != other.dims(); }
};

/** Create a new shape using a list of `DimIndices...` to use as the
 * dimensions of the shape. The new shape's i'th dimension will be the
 * j'th dimension of `shape` where j is the i'th value of `DimIndices...`.
 *
 * `transpose` requires `DimIndices...` to be a permutation, while
 * `reorder` accepts a list of indices that may be a subset of the
 * dimensions.
 *
 * Examples:
 * - `transpose<2, 0, 1>(s_3d) == make_shape(s.z(), s.y(), s.x())`
 * - `reorder<1, 2>(s_4d) == make_shape(s.y(), s.z())` */
template <size_t... DimIndices, class... Dims,
    class = internal::enable_if_permutation<sizeof...(Dims), DimIndices...>>
auto transpose(const shape<Dims...>& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}
template <size_t... DimIndices, class... Dims>
auto reorder(const shape<Dims...>& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}

namespace internal {

template<size_t D, class Dims, class Fn, class... Indices,
    std::enable_if_t<(D == 0), int> = 0>
void for_each_index_in_order(const Dims& dims, Fn&& fn, const std::tuple<Indices...>& indices) {
  for (index_t i : std::get<D>(dims)) {
    fn(std::tuple_cat(std::make_tuple(i), indices));
  }
}

template<size_t D, class Dims, class Fn, class... Indices,
    std::enable_if_t<(D > 0), int> = 0>
void for_each_index_in_order(const Dims& dims, Fn&& fn, const std::tuple<Indices...>& indices) {
  for (index_t i : std::get<D>(dims)) {
    for_each_index_in_order<D - 1>(dims, fn, std::tuple_cat(std::make_tuple(i), indices));
  }
}

template <size_t D>
NDARRAY_INLINE void advance() {}
template <size_t D, class Ptr, class... Ptrs>
NDARRAY_INLINE void advance(Ptr& ptr, Ptrs&... ptrs) {
  std::get<0>(ptr) += std::get<D>(std::get<1>(ptr));
  advance<D>(ptrs...);
}

template <size_t D, class ExtentType, class Fn, class... Ptrs,
    std::enable_if_t<(D == 0), int> = 0>
void for_each_value_in_order(const ExtentType& extent, Fn&& fn, Ptrs... ptrs) {
  index_t extent_d = std::get<D>(extent);
  if (all(std::get<D>(std::get<1>(ptrs)) == 1 ...)) {
    for (index_t i = 0; i < extent_d; i++) {
      fn(*std::get<0>(ptrs)++...);
    }
  } else {
    for (index_t i = 0; i < extent_d; i++) {
      fn(*std::get<0>(ptrs)...);
      advance<D>(ptrs...);
    }
  }
}

template <size_t D, class ExtentType, class Fn, class... Ptrs,
    std::enable_if_t<(D > 0), int> = 0>
void for_each_value_in_order(const ExtentType& extent, Fn&& fn, Ptrs... ptrs) {
  index_t extent_d = std::get<D>(extent);
  for (index_t i = 0; i < extent_d; i++) {
    for_each_value_in_order<D - 1>(extent, fn, ptrs...);
    advance<D>(ptrs...);
  }
}

// Scalar buffers are a special case.
template <size_t D, class Fn, class... Ptrs>
void for_each_value_in_order(const std::tuple<>& extent, Fn&& fn, Ptrs... ptrs) {
  fn(*std::get<0>(ptrs)...);
}

template <typename TSrc, typename TDst>
NDARRAY_INLINE void move_assign(TSrc& src, TDst& dst) {
  dst = std::move(src);
}

template <typename TSrc, typename TDst>
NDARRAY_INLINE void copy_assign(const TSrc& src, TDst& dst) {
  dst = src;
}

template <size_t Rank, std::enable_if_t<(Rank > 0), int> = 0>
auto make_default_dense_shape() {
  return make_shape_from_tuple(std::tuple_cat(
      std::make_tuple(dense_dim<>()), typename tuple_of_n<dim<>, Rank - 1>::type()));
}
template <size_t Rank, std::enable_if_t<(Rank == 0), int> = 0>
auto make_default_dense_shape() {
  return shape<>();
}

template <class Shape, size_t... Is>
auto make_dense_shape(const Shape& dims, index_sequence<Is...>) {
  return make_shape(
      dense_dim<>(std::get<0>(dims).min(), std::get<0>(dims).extent()),
      dim<>(std::get<Is + 1>(dims).min(), std::get<Is + 1>(dims).extent())...);
}

template <class Shape, size_t... Is>
Shape without_strides(const Shape& s, index_sequence<Is...>) {
  return {{s.template dim<Is>().min(), s.template dim<Is>().extent()}...};
}

template <index_t Min, index_t Extent, index_t Stride, class DimSrc>
bool is_dim_compatible(const dim<Min, Extent, Stride>&, const DimSrc& src) {
  return
    (is_dynamic(Min) || src.min() == Min) &&
    (is_dynamic(Extent) || src.extent() == Extent) &&
    (is_dynamic(Stride) || src.stride() == Stride);
}

template <class... DimsDst, class ShapeSrc, size_t... Is>
bool is_shape_compatible(const shape<DimsDst...>&, const ShapeSrc& src, index_sequence<Is...>) {
  return all(is_dim_compatible(DimsDst(), src.template dim<Is>())...);
}

template <class DimA, class DimB>
auto clamp_dims(const DimA& a, const DimB& b) {
  constexpr index_t Min = static_max(DimA::Min, DimB::Min);
  constexpr index_t Max = static_min(DimA::Max, DimB::Max);
  constexpr index_t Extent = static_add(static_sub(Max, Min), 1);
  index_t min = std::max(a.min(), b.min());
  index_t max = std::min(a.max(), b.max());
  index_t extent = max - min + 1;
  return dim<Min, Extent, DimA::Stride>(min, extent, a.stride());
}

template <class DimsA, class DimsB, size_t... Is>
auto clamp(const DimsA& a, const DimsB& b, index_sequence<Is...>) {
  return make_shape(clamp_dims(std::get<Is>(a), std::get<Is>(b))...);
}

// Shuffle a tuple with indices Is...
template <size_t... Is, class T>
auto shuffle(const T& t) {
  return std::make_tuple(std::get<Is>(t)...);
}

// Return where the index I appears in Is...
template <size_t I>
constexpr size_t index_of() { return 10000; }
template <size_t I, size_t I0, size_t... Is>
constexpr size_t index_of() {
  return I == I0 ? 0 : 1 + index_of<I, Is...>();
}

// Similar to std::get, but returns a one-element tuple if I is
// in bounds, or an empty tuple if not.
template <size_t I, class T, std::enable_if_t<(I < std::tuple_size<T>::value), int> = 0>
NDARRAY_INLINE auto get_tuple(const T& t) {
  return std::make_tuple(std::get<I>(t));
}
template <size_t I, class T, std::enable_if_t<(I >= std::tuple_size<T>::value), int> = 0>
NDARRAY_INLINE auto get_tuple(const T& t) {
  return std::make_tuple();
}

// Perform the inverse of a shuffle with indices Is...
template <size_t... Is, class T, size_t... Js>
auto unshuffle(const T& t, index_sequence<Js...>) {
  return std::tuple_cat(get_tuple<index_of<Js, Is...>()>(t)...);
}
template <size_t... Is, class... Ts>
auto unshuffle(const std::tuple<Ts...>& t) {
  return unshuffle<Is...>(t, make_index_sequence<variadic_max(Is...) + 1>());
}

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_compatible =
    typename std::enable_if<std::is_constructible<ShapeDst, ShapeSrc>::value>::type;

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_explicitly_compatible =
    typename std::enable_if<(ShapeSrc::rank() <= ShapeSrc::rank())>::type;

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_copy_compatible =
    typename std::enable_if<(ShapeDst::rank() == ShapeSrc::rank())>::type;

}  // namespace internal

/** An arbitrary `shape` with the specified rank `Rank`. This shape is
 * compatible with any other shape of the same rank. */
template <size_t Rank>
using shape_of_rank =
    decltype(make_shape_from_tuple(typename internal::tuple_of_n<dim<>, Rank>::type()));

/** A `shape` where the innermost dimension is a `dense_dim`, and all other
 * dimensions are arbitrary. */
template <size_t Rank>
using dense_shape = decltype(internal::make_default_dense_shape<Rank>());

/** Make a `dense_shape` with the same mins and extents as `s`. */
template <class... Dims>
auto make_dense(const shape<Dims...>& s) {
  constexpr size_t rank = sizeof...(Dims);
  return internal::make_dense_shape(s.dims(), internal::make_index_sequence<rank - 1>());
}
inline auto make_dense(const shape<>& s) { return s; }

/** Replace the strides of `s` with minimal strides, as determined by
 * the `shape::resolve` algorithm. The strides of `s` are replaced with
 * a possibly different order, even if the shape is already compact.
 *
 * The resulting shape may not have `Shape::is_compact` return `true`
 * if the shape has non-compact compile-time constant strides. */
template <class Shape>
Shape make_compact(const Shape& s) {
  Shape without_strides =
      internal::without_strides(s, internal::make_index_sequence<Shape::rank()>());
  without_strides.resolve();
  return without_strides;
}

/** Returns `true` if a shape `src` can be assigned to a shape of type
 * `ShapeDst` without error. */
template <class ShapeDst, class ShapeSrc,
    class = internal::enable_if_shapes_compatible<ShapeSrc, ShapeDst>>
bool is_compatible(const ShapeSrc& src) {
  return internal::is_shape_compatible(
      ShapeDst(), src, internal::make_index_sequence<ShapeSrc::rank()>());
}

/** Convert a shape `src` to shape type `ShapeDst`. This explicit conversion
 * allows converting a low rank shape to a higher ranked shape, where new
 * dimensions have min 0 and extent 1. */
// TODO: Consider enabling this kind of conversion implicitly. It is hard to
// do without constructor overload ambiguity problems, and I'm also not sure
// it's a good idea.
template <class ShapeDst, class ShapeSrc,
    class = internal::enable_if_shapes_explicitly_compatible<ShapeDst, ShapeSrc>>
ShapeDst convert_shape(const ShapeSrc& src) {
  return internal::convert_dims<typename ShapeDst::dims_type>(
      src.dims(), std::make_index_sequence<ShapeDst::rank()>());
}

/** Test if a shape `src` can be explicitly converted to a shape of type
 * `ShapeDst` using `convert_shape` without error. */
template <class ShapeDst, class ShapeSrc,
    class = internal::enable_if_shapes_explicitly_compatible<ShapeSrc, ShapeDst>>
bool is_explicitly_compatible(const ShapeSrc& src) {
  return internal::is_shape_compatible(
      ShapeDst(), src, internal::make_index_sequence<ShapeSrc::rank()>());
}

/** Iterate over all indices in the shape, calling a function `fn` for each set
 * of indices. The indices are in the same order as the dims in the shape. The
 * first dim is the 'inner' loop of the iteration, and the last dim is the
 * 'outer' loop.
 *
 * These functions are typically used to implement `shape_traits<>` and
 * `copy_shape_traits<>` objects. Use `for_each_index`,
 * `array_ref<>::for_each_value`, or `array<>::for_each_value` instead. */
template<class Shape, class Fn,
    class = internal::enable_if_callable<Fn, typename Shape::index_type>>
void for_each_index_in_order(const Shape& shape, Fn &&fn) {
  internal::for_each_index_in_order<Shape::rank() - 1>(shape.dims(), fn, std::tuple<>());
}
template<class Shape, class Ptr, class Fn,
    class = internal::enable_if_callable<Fn, typename std::remove_pointer<Ptr>::type&>>
void for_each_value_in_order(const Shape& shape, Ptr base, Fn &&fn) {
  using index_type = typename Shape::index_type;
  // TODO: This is losing compile-time constant extents and strides info
  // (https://github.com/dsharlet/array/issues/1).
  std::tuple<Ptr, index_type> base_and_stride(base, shape.stride());
  internal::for_each_value_in_order<Shape::rank() - 1>(shape.extent(), fn, base_and_stride);
}

/** Similar to `for_each_value_in_order`, but iterates over two arrays
 * simultaneously. `shape` defines the loop nest, while `shape_a` and `shape_b`
 * define the memory layout of `base_a` and `base_b`. */
template<class Shape, class ShapeA, class PtrA, class ShapeB, class PtrB, class Fn,
    class = internal::enable_if_callable<Fn,
        typename std::remove_pointer<PtrA>::type&, typename std::remove_pointer<PtrB>::type&>>
void for_each_value_in_order(
    const Shape& shape, const ShapeA& shape_a, PtrA base_a, const ShapeB& shape_b, PtrB base_b,
    Fn &&fn) {
  base_a += shape_a(shape.min());
  base_b += shape_b(shape.min());
  // TODO: This is losing compile-time constant extents and strides info
  // (https://github.com/dsharlet/array/issues/1).
  using index_type = typename Shape::index_type;
  std::tuple<PtrA, index_type> a(base_a, shape_a.stride());
  std::tuple<PtrB, index_type> b(base_b, shape_b.stride());
  internal::for_each_value_in_order<Shape::rank() - 1>(shape.extent(), fn, a, b);
}

namespace internal {

inline bool can_fuse(const dim<>& inner, const dim<>& outer) {
  return inner.stride() * inner.extent() == outer.stride();
}

inline dim<> fuse(const dim<>& inner, const dim<>& outer) {
  assert(can_fuse(inner, outer));
  return dim<>(
      inner.min() + outer.min() * inner.extent(),
      inner.extent() * outer.extent(),
      inner.stride());
}

// We need a sort that only needs to deal with very small lists,
// and extra complexity here is costly in code size/compile time.
// This is a rare job for bubble sort!
template <class Iterator, class Compare>
void bubble_sort(Iterator begin, Iterator end, Compare&& comp) {
  for (Iterator i = begin; i != end; ++i) {
    for (Iterator j = i; j != end; ++j) {
      if (comp(*j, *i)) {
        std::swap(*i, *j);
      }
    }
  }
}

// Sort the dims such that strides are increasing from dim 0, and contiguous
// dimensions are fused.
template <class Shape>
shape_of_rank<Shape::rank()> dynamic_optimize_shape(const Shape& shape) {
  auto dims = internal::tuple_to_array<dim<>>(shape.dims());

  // Sort the dims by stride.
  bubble_sort(dims.begin(), dims.end(), [](const dim<>& l, const dim<>& r) {
    return l.stride() < r.stride();
  });

  // Find dimensions that are contiguous and fuse them.
  size_t rank = dims.size();
  for (size_t i = 0; i + 1 < rank;) {
    if (can_fuse(dims[i], dims[i + 1])) {
      dims[i] = fuse(dims[i], dims[i + 1]);
      for (size_t j = i + 1; j + 1 < rank; j++) {
        dims[j] = dims[j + 1];
      }
      rank--;
    } else {
      i++;
    }
  }

  // Unfortunately, we can't make the rank of the resulting shape smaller. Fill
  // the end of the array with size 1 dimensions.
  for (size_t i = rank; i < dims.size(); i++) {
    dims[i] = dim<>(0, 1, dims[i - 1].stride() * dims[i - 1].extent());
  }

  return shape_of_rank<Shape::rank()>(array_to_tuple(dims));
}

// Optimize a src and dst shape. The dst shape is made dense, and contiguous
// dimensions are fused.
template <class ShapeSrc, class ShapeDst,
    class = enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
auto dynamic_optimize_copy_shapes(const ShapeSrc& src, const ShapeDst& dst) {
  constexpr size_t rank = ShapeSrc::rank();
  static_assert(rank == ShapeDst::rank(), "copy shapes must have same rank.");
  auto src_dims = internal::tuple_to_array<dim<>>(src.dims());
  auto dst_dims = internal::tuple_to_array<dim<>>(dst.dims());

  struct copy_dims {
    dim<> src;
    dim<> dst;
  };
  std::array<copy_dims, rank> dims;
  for (size_t i = 0; i < rank; i++) {
    dims[i] = {src_dims[i], dst_dims[i]};
  }

  // Sort the dims by the dst stride.
  bubble_sort(dims.begin(), dims.end(), [](const copy_dims& l, const copy_dims& r) {
    return l.dst.stride() < r.dst.stride();
  });

  // Find dimensions that are contiguous and fuse them.
  size_t new_rank = dims.size();
  for (size_t i = 0; i + 1 < new_rank;) {
    if (dims[i].src.extent() == dims[i].dst.extent() &&
        can_fuse(dims[i].src, dims[i + 1].src) &&
        can_fuse(dims[i].dst, dims[i + 1].dst)) {
      dims[i].src = fuse(dims[i].src, dims[i + 1].src);
      dims[i].dst = fuse(dims[i].dst, dims[i + 1].dst);
      for (size_t j = i + 1; j + 1 < new_rank; j++) {
        dims[j] = dims[j + 1];
      }
      new_rank--;
    } else {
      i++;
    }
  }

  // Unfortunately, we can't make the rank of the resulting shape dynamic. Fill
  // the end of the array with size 1 dimensions.
  for (size_t i = new_rank; i < dims.size(); i++) {
    dims[i] = {
        dim<>(0, 1, dims[i - 1].src.stride() * dims[i - 1].src.extent()),
        dim<>(0, 1, dims[i - 1].dst.stride() * dims[i - 1].dst.extent())};
  }

  for (size_t i = 0; i < dims.size(); i++) {
    src_dims[i] = dims[i].src;
    dst_dims[i] = dims[i].dst;
  }

  return std::make_pair(
      shape_of_rank<rank>(array_to_tuple(src_dims)),
      shape_of_rank<rank>(array_to_tuple(dst_dims)));
}

template <class Shape>
auto optimize_shape(const Shape& shape) {
  // In the general case, dynamically optimize the shape.
  return dynamic_optimize_shape(shape);
}

template <class Dim0>
auto optimize_shape(const shape<Dim0>& shape) {
  // Nothing to do for rank 1 shapes.
  return shape;
}

template <class ShapeSrc, class ShapeDst>
auto optimize_copy_shapes(const ShapeSrc& src, const ShapeDst& dst) {
  return dynamic_optimize_copy_shapes(src, dst);
}

template <class Dim0Src, class Dim0Dst>
auto optimize_copy_shapes(const shape<Dim0Src>& src, const shape<Dim0Dst>& dst) {
  // Nothing to do for rank 1 shapes.
  return std::make_pair(src, dst);
}

template <class T>
T* pointer_add(T* x, index_t offset) {
  return x != nullptr ? x + offset : x;
}

}  // namespace internal

/** Shape traits enable some behaviors to be customized per shape type. */
template <class Shape>
class shape_traits {
 public:
  using shape_type = Shape;

  /** The `for_each_index` implementation for the shape may choose to iterate in a
   * different order than the default (in-order). */
  template <class Fn>
  static void for_each_index(const Shape& shape, Fn&& fn) {
    for_each_index_in_order(shape, fn);
  }

  /** The `for_each_value` implementation for the shape may be able to statically
   * optimize shape. The default implementation optimizes the shape at runtime,
   * and only attempts to convert the shape to a `dense_shape`. */
  template <class Ptr, class Fn>
  static void for_each_value(const Shape& shape, Ptr base, Fn&& fn) {
    auto opt_shape = internal::optimize_shape(shape);
    for_each_value_in_order(opt_shape, base, fn);
  }
};

template <>
class shape_traits<shape<>> {
 public:
  using shape_type = shape<>;

  template <class Fn>
  static void for_each_index(const shape<>&, Fn&& fn) {
    fn(std::tuple<>());
  }

  template <class Ptr, class Fn>
  static void for_each_value(const shape<>&, Ptr base, Fn&& fn) {
    fn(*base);
  }
};

/** Copy shape traits enable some behaviors to be customized on a pairwise shape
 * basis for copies. */
template <class ShapeSrc, class ShapeDst>
class copy_shape_traits {
 public:
  using src_shape_type = ShapeSrc;
  using dst_shape_type = ShapeDst;

  /** The `for_each_value` implementation for the shapes may be able to statically
   * optimize the shapes. The default implementation optimizes the shapes at
   * runtime, and only attempts to convert the shapes to `dense_shape`s of the
   * same rank. */
  template <class Fn, class TSrc, class TDst>
  static void for_each_value(
      const ShapeSrc& shape_src, TSrc src, const ShapeDst& shape_dst, TDst dst, Fn&& fn) {
    // For this function, we don't care about the order in which the callback is
    // called. Optimize the shapes for memory access order.
    auto opt_shape = internal::optimize_copy_shapes(shape_src, shape_dst);
    const auto& opt_shape_src = opt_shape.first;
    const auto& opt_shape_dst = opt_shape.second;

    for_each_value_in_order(opt_shape_dst, opt_shape_src, src, opt_shape_dst, dst, fn);
  }
};

/** Iterate over all indices in the shape `s`, calling a function `fn` for
 * each set of indices. `for_all_indices` calls `fn` with a list of
 * arguments corresponding to each dim. `for_each_index` calls `fn` with an
 * index tuple describing the indices.
 *
 * If the `LoopOrder...` permutation is empty, the order of the loops is
 * defined by `shape_traits<Shape>`, and the callable `fn` must accept
 * a `Shape::index_type` in the case of `for_each_index`, or `Shape::rank()`
 * `index_t` objects in the case of `for_all_indices`.
 *
 * If the `LoopOrder...` permutation is not empty, the order of the loops is
 * defined by this ordering. The first index of `LoopOrder...` is the innermost
 * loop of the loop nest. The callable `fn` must accept an
 * `index_of_rank<sizeof...(LoopOprder)>` in the case of `for_each_index<>`,
 * or `sizeof...(LoopOrder)` `index_t` objects in the case of
 * `for_all_indices<>`.  */
template <size_t... LoopOrder, class Shape, class Fn,
    class = internal::enable_if_callable<Fn, typename Shape::index_type>,
    std::enable_if_t<(sizeof...(LoopOrder) == 0), int> = 0>
void for_each_index(const Shape& s, Fn&& fn) {
  shape_traits<Shape>::for_each_index(s, fn);
}
template <size_t... LoopOrder, class Shape, class Fn,
    class = internal::enable_if_applicable<Fn, typename Shape::index_type>,
    std::enable_if_t<(sizeof...(LoopOrder) == 0), int> = 0>
void for_all_indices(const Shape& s, Fn&& fn) {
  using index_type = typename Shape::index_type;
  for_each_index(s, [&](const index_type&i) {
    internal::apply(fn, i);
  });
}
template <size_t... LoopOrder, class Shape, class Fn,
    class = internal::enable_if_callable<Fn, index_of_rank<sizeof...(LoopOrder)>>,
    std::enable_if_t<(sizeof...(LoopOrder) != 0), int> = 0>
void for_each_index(const Shape& s, Fn&& fn) {
  using index_type = index_of_rank<sizeof...(LoopOrder)>;
  for_each_index_in_order(reorder<LoopOrder...>(s), [&](const index_type& i) {
    fn(internal::unshuffle<LoopOrder...>(i));
  });
}
template <size_t... LoopOrder, class Shape, class Fn,
    class = internal::enable_if_callable<Fn, decltype(LoopOrder)...>,
    std::enable_if_t<(sizeof...(LoopOrder) != 0), int> = 0>
void for_all_indices(const Shape& s, Fn&& fn) {
  using index_type = index_of_rank<sizeof...(LoopOrder)>;
  for_each_index_in_order(reorder<LoopOrder...>(s), [&](const index_type&i) {
    internal::apply(fn, internal::unshuffle<LoopOrder...>(i));
  });
}

template <class T, class Shape>
class array_ref;
template <class T, class Shape, class Alloc>
class array;

template <class T, class Shape>
using const_array_ref = array_ref<const T, Shape>;

/** Make a new `array_ref` with shape `shape` and base pointer `base`. */
template <class T, class Shape>
array_ref<T, Shape> make_array_ref(T* base, const Shape& shape) {
  return {base, shape};
}

namespace internal {

template <class T, class Shape, class... Args>
auto make_array_ref_at(T base, const Shape& shape, const std::tuple<Args...>& args) {
  auto new_shape = shape(args);
  auto new_mins =
      mins_of_intervals(args, shape.dims(), make_index_sequence<sizeof...(Args)>());
  auto old_min_offset = shape(new_mins);
  return make_array_ref(internal::pointer_add(base, old_min_offset), new_shape);
}

}  // namespace internal

/** A reference to an array is an object with a shape mapping indices to flat
 * offsets, which are used to dereference a pointer. This object does not own
 * any memory, and it is cheap to copy. */
template <class T, class Shape>
class array_ref {
 public:
  /** Type of elements referenced in this array_ref. */
  using value_type = T;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  /** Type of the shape of this array_ref. */
  using shape_type = Shape;
  using index_type = typename Shape::index_type;
  using shape_traits_type = shape_traits<Shape>;
  using size_type = size_t;

  /** The number of dims in the shape of this array. */
  static constexpr size_t rank() { return Shape::rank(); }

  /** True if the rank of this array is 0. */
  static constexpr bool is_scalar() { return Shape::is_scalar(); }

 private:
  template <class OtherShape>
  using enable_if_shape_compatible = internal::enable_if_shapes_compatible<Shape, OtherShape>;

  template <class... Args>
  using enable_if_same_rank = typename std::enable_if<sizeof...(Args) == rank()>::type;

  template <class... Args>
  using enable_if_indices =
      typename std::enable_if<internal::all_of_type<index_t, Args...>::value>::type;

  template <class... Args>
  using enable_if_slices = typename std::enable_if<
      internal::all_of_type<interval<>, Args...>::value &&
      !internal::all_of_type<index_t, Args...>::value>::type;

  template <size_t Dim>
  using enable_if_dim = typename std::enable_if<Dim < rank()>::type;

  pointer base_;
  Shape shape_;

 public:
  /** Make an array_ref to the given `base` pointer, interpreting it as having
   * the shape `shape`. */
  array_ref(pointer base = nullptr, const Shape& shape = Shape()) : base_(base), shape_(shape) {
    shape_.resolve();
  }

  /** Shallow copy or assign an array_ref. */
  array_ref(const array_ref& other) = default;
  array_ref(array_ref&& other) = default;
  array_ref& operator=(const array_ref& other) = default;
  array_ref& operator=(array_ref&& other) = default;

  /** Shallow copy or assign an array_ref with a different shape type. */
  template <class OtherShape, class = enable_if_shape_compatible<OtherShape>>
  array_ref(const array_ref<T, OtherShape>& other) : array_ref(other.base(), other.shape()) {}
  template <class OtherShape, class = enable_if_shape_compatible<OtherShape>>
  array_ref& operator=(const array_ref<T, OtherShape>& other) {
    base_ = other.base();
    shape_ = other.shape();
    return *this;
  }

  /** Get a reference to the element at `indices`. */
  reference operator() (const index_type& indices) const { return base_[shape_(indices)]; }
  reference operator[] (const index_type& indices) const { return base_[shape_(indices)]; }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  reference operator() (Args... indices) const { return base_[shape_(indices...)]; }

  /** Create an `array_ref` from this array_ref using a indices or intervals
   * `args`. Dimensions corresponding to indices in `args` are sliced, i.e.
   * the result will not have this dimension. The rest of the dimensions are
   * cropped. */
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator[] (const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (Args... args) const {
    return internal::make_array_ref_at(base_, shape_, std::make_tuple(args...));
  }

  /** Call a function with a reference to each value in this array_ref. The
   * order in which `fn` is called is undefined, to enable optimized memory
   * access patterns. */
  template <class Fn, class = internal::enable_if_callable<Fn, reference>>
  void for_each_value(Fn&& fn) const {
    shape_traits_type::for_each_value(shape_, base_, fn);
  }

  /** Pointer to the element at the min index of the shape. */
  pointer base() const { return base_; }

  /** Pointer to the element at the beginning of the flat array. */
  pointer data() const { return internal::pointer_add(base_, shape_.flat_min()); }

  /** Shape of this array_ref. */
  const Shape& shape() const { return shape_; }

  template <size_t D, class = enable_if_dim<D>>
  auto& dim() { return shape_.template dim<D>(); }
  template <size_t D, class = enable_if_dim<D>>
  const auto& dim() const { return shape_.template dim<D>(); }
  size_type size() const { return shape_.size(); }
  bool empty() const { return base() != nullptr ? shape_.empty() : true; }
  bool is_compact() const { return shape_.is_compact(); }

  /** Provide some aliases for common interpretations of dimensions
   * `i`, `j`, `k` as dimensions 0, 1, 2, respectively. */
  auto& i() { return shape_.i(); }
  const auto& i() const { return shape_.i(); }
  auto& j() { return shape_.j(); }
  const auto& j() const { return shape_.j(); }
  auto& k() { return shape_.k(); }
  const auto& k() const { return shape_.k(); }

  /** Provide some aliases for common interpretations of dimensions
   * `x`, `y`, `z` or `c`, `w` as dimensions 0, 1, 2, 3 respectively. */
  auto& x() { return shape_.x(); }
  const auto& x() const { return shape_.x(); }
  auto& y() { return shape_.y(); }
  const auto& y() const { return shape_.y(); }
  auto& z() { return shape_.z(); }
  const auto& z() const { return shape_.z(); }
  auto& c() { return shape_.c(); }
  const auto& c() const { return shape_.c(); }
  auto& w() { return shape_.w(); }
  const auto& w() const { return shape_.w(); }

  /** Assuming this array represents an image with dimensions width, height,
   * channels, get the extent of those dimensions. */
  index_t width() const { return shape_.width(); }
  index_t height() const { return shape_.height(); }
  index_t channels() const { return shape_.channels(); }

  /** Assuming this array represents a matrix with dimensions {rows, cols}, get
   * the extent of those dimensions. */
  index_t rows() const { return shape_.rows(); }
  index_t columns() const { return shape_.columns(); }

  /** Compare the contents of this array_ref to `other`. For two array_refs to
   * be considered equal, they must have the same shape, and all elements
   * addressable by the shape must also be equal. */
  // TODO: Maybe this should just check for equality of the shape and pointer,
  // and let the free function equal serve this purpose.
  bool operator!=(const array_ref& other) const {
    if (shape_ != other.shape_) {
      return true;
    }

    // TODO: This currently calls operator!= on all elements of the array_ref,
    // even after we find a non-equal element
    // (https://github.com/dsharlet/array/issues/4).
    bool result = false;
    copy_shape_traits<Shape, Shape>::for_each_value(
        shape_, base_, other.shape_, other.base_, [&](const_reference a, const_reference b) {
      if (a != b) {
        result = true;
      }
    });
    return result;
  }
  bool operator==(const array_ref& other) const { return !operator!=(other); }

  const array_ref<T, Shape>& ref() const { return *this; }

  /** Allow conversion from array_ref<T> to const_array_ref<T>. */
  const const_array_ref<T, Shape> cref() const { return const_array_ref<T, Shape>(base_, shape_); }
  operator const_array_ref<T, Shape>() const { return cref(); }

  /** Change the shape of the array to `new_shape`, and move the base pointer
   * by `offset`. The new shape must be a subset of the old shape. */
  void set_shape(const Shape& new_shape, index_t offset = 0) {
    assert(new_shape.is_resolved());
    assert(new_shape.is_subset_of(shape_, -offset));
    shape_ = new_shape;
    base_ = internal::pointer_add(base_, offset);
  }
};

/** array_ref with an arbitrary shape of `Rank`. */
template <class T, size_t Rank>
using array_ref_of_rank = array_ref<T, shape_of_rank<Rank>>;
template <class T, size_t Rank>
using const_array_ref_of_rank = array_ref_of_rank<const T, Rank>;

/** array_ref with a shape `dense_shape<Rank>`. */
template <class T, size_t Rank>
using dense_array_ref = array_ref<T, dense_shape<Rank>>;
template <class T, size_t Rank>
using const_dense_array_ref = dense_array_ref<const T, Rank>;

/** A multi-dimensional array container that owns an allocation of memory. */
template <class T, class Shape, class Alloc = std::allocator<T>>
class array {
 public:
  /** Type of the allocator used to allocate memory in this array. */
  using allocator_type = Alloc;
  using alloc_traits = std::allocator_traits<Alloc>;

  /** Type of the values stored in this array. */
  using value_type = T;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = typename alloc_traits::pointer;
  using const_pointer = typename alloc_traits::const_pointer;

  /** Type of the shape of this array. */
  using shape_type = Shape;
  using index_type = typename Shape::index_type;
  using shape_traits_type = shape_traits<Shape>;
  using copy_shape_traits_type = copy_shape_traits<Shape, Shape>;
  using size_type = size_t;

  /** The number of dims in the shape of this array. */
  static constexpr size_t rank() { return Shape::rank(); }

  /** True if the rank of this array is 0. */
  static constexpr bool is_scalar() { return Shape::is_scalar(); }

 private:
  template <class... Args>
  using enable_if_same_rank = typename std::enable_if<sizeof...(Args) == rank()>::type;

  template <class... Args>
  using enable_if_indices =
      typename std::enable_if<internal::all_of_type<index_t, Args...>::value>::type;

  template <class... Args>
  using enable_if_slices = typename std::enable_if<
      internal::all_of_type<interval<>, Args...>::value &&
      !internal::all_of_type<index_t, Args...>::value>::type;

  template <size_t Dim>
  using enable_if_dim = typename std::enable_if<Dim < rank()>::type;

  Alloc alloc_;
  pointer buffer_;
  size_type buffer_size_;
  pointer base_;
  Shape shape_;

  // After allocate the array is allocated but uninitialized.
  void allocate() {
    assert(!buffer_);
    shape_.resolve();
    size_type flat_extent = shape_.flat_extent();
    if (flat_extent > 0) {
      buffer_size_ = flat_extent;
      buffer_ = alloc_traits::allocate(alloc_, buffer_size_);
    }
    base_ = buffer_ - shape_.flat_min();
  }

  // Call the constructor on all of the elements of the array.
  void construct() {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) {
      alloc_traits::construct(alloc_, &x);
    });
  }
  void construct(const T& init) {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) {
      alloc_traits::construct(alloc_, &x, init);
    });
  }
  void copy_construct(const array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape());
    copy_shape_traits_type::for_each_value(
        other.shape(), other.base(), shape_, base_, [&](const_reference src, reference dst) {
      alloc_traits::construct(alloc_, &dst, src);
    });
  }
  void move_construct(array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape());
    copy_shape_traits_type::for_each_value(
        other.shape(), other.base(), shape_, base_, [&](reference src, reference dst) {
      alloc_traits::construct(alloc_, &dst, std::move(src));
    });
  }

  // Call the dstructor on every element.
  void destroy() {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) {
      alloc_traits::destroy(alloc_, &x);
    });
  }

  void deallocate() {
    if (base_) {
      destroy();
      base_ = nullptr;
      shape_ = Shape();
      alloc_traits::deallocate(alloc_, buffer_, buffer_size_);
      buffer_ = nullptr;
    }
  }

 public:
  /** Construct an array with a default constructed Shape. Most shapes are empty
   * by default, but a Shape with non-zero compile-time constants for all
   * extents will be non-empty. */
  array() : array(Shape()) {}
  explicit array(const Alloc& alloc) : array(Shape(), alloc) {}

  /** Construct an array with a particular `shape`, allocated by `alloc`. All
   * elements in the array are copy-constructed from `value`. */
  array(const Shape& shape, const T& value, const Alloc& alloc = Alloc()) : array(alloc) {
    assign(shape, value);
  }

  /** Construct an array with a particular `shape`, allocated by `alloc`, with
   * default constructed elements. */
  explicit array(const Shape& shape, const Alloc& alloc = Alloc())
      : alloc_(alloc), buffer_(nullptr), buffer_size_(0), base_(nullptr), shape_(shape) {
    allocate();
    construct();
  }

  /** Copy construct from another array `other`, using copy's allocator. This is
   * a deep copy of the contents of `other`. */
  array(const array& other)
      : array(alloc_traits::select_on_container_copy_construction(other.get_allocator())) {
    assign(other);
  }

  /** Copy construct from another array `other`. The array is allocated using
   * `alloc`. This is a deep copy of the contents of `other`. */
  array(const array& other, const Alloc& alloc) : array(alloc) {
    assign(other);
  }

  /** Move construct from another array `other`. If the allocator of this array
   * and the other array are equal, this operation moves the allocation of other
   * to this array, and the other array becomes a default constructed array. If
   * the allocator of this and the other array are non-equal, each element is
   * move-constructed into a new allocation. */
  array(array&& other) : array(std::move(other), Alloc()) {}
  array(array&& other, const Alloc& alloc)
      : alloc_(alloc), buffer_(nullptr), buffer_size_(0), base_(nullptr) {
    if (alloc_ != other.get_allocator()) {
      shape_ = other.shape_;
      allocate();
      move_construct(other);
    } else {
      using std::swap;
      swap(buffer_, other.buffer_);
      swap(buffer_size_, other.buffer_size_);
      swap(base_, other.base_);
      swap(shape_, other.shape_);
    }
  }
  ~array() {
    deallocate();
  }

  // Let's choose not to provide array(const array_ref&) constructors. This is
  // a deep copy that may be unintentional, perhaps it is better to require
  // being explicit via `make_copy`.

  /** Assign the contents of the array as a copy of `other`. The array is
   * deallocated if the allocator cannot be propagated on assignment. The array
   * is then reallocated if necessary, and each element in the array is copy
   * constructed from other. */
  array& operator=(const array& other) {
    if (base_ == other.base()) {
      if (base_) {
        assert(shape_ == other.shape());
      } else {
        shape_ = other.shape();
        assert(shape_.empty());
      }
      return *this;
    }

    if (alloc_traits::propagate_on_container_copy_assignment::value) {
      if (alloc_ != other.get_allocator()) {
        deallocate();
      }
      alloc_ = other.get_allocator();
    }

    assign(other);
    return *this;
  }

  /** Assign the contents of the array by moving from `other`. If the allocator
   * can be propagated on move assignment, the allocation of `other` is moved in
   * an O(1) operation. If the allocator cannot be propagated, each element is
   * move-assigned from `other`. */
  array& operator=(array&& other) {
    using std::swap;
    if (base_ == other.base()) {
      if (base_) {
        assert(shape_ == other.shape());
      } else {
        swap(shape_, other.shape_);
        assert(shape_.empty());
      }
      return *this;
    }

    if (std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value) {
      swap(alloc_, other.alloc_);
      swap(buffer_, other.buffer_);
      swap(buffer_size_, other.buffer_size_);
      swap(base_, other.base_);
      swap(shape_, other.shape_);
    } else if (alloc_ == other.get_allocator()) {
      swap(buffer_, other.buffer_);
      swap(buffer_size_, other.buffer_size_);
      swap(base_, other.base_);
      swap(shape_, other.shape_);
    } else {
      assign(std::move(other));
    }
    return *this;
  }

  /** Assign the contents of the array to be a copy or move of `other`. The
   * array is destroyed, reallocated if necessary, and then each element is
   * copy- or move-constructed from `other`. */
  void assign(const array& other) {
    if (base_ == other.base()) {
      if (base_) {
        assert(shape_ == other.shape());
      } else {
        shape_ = other.shape();
        assert(shape_.empty());
      }
      return;
    }
    if (shape_ == other.shape()) {
      destroy();
    } else {
      deallocate();
      shape_ = other.shape();
      allocate();
    }
    copy_construct(other);
  }
  void assign(array&& other) {
    if (base_ == other.base()) {
      if (base_) {
        assert(shape_ == other.shape());
      } else {
        shape_ = other.shape();
        assert(shape_.empty());
      }
      return;
    }
    if (shape_ == other.shape()) {
      destroy();
    } else {
      deallocate();
      shape_ = other.shape();
      allocate();
    }
    move_construct(other);
  }

  /** Assign the contents of this array to have `shape` with each element
   * copy constructed from `value`. */
  void assign(Shape shape, const T& value) {
    shape.resolve();
    if (shape_ == shape) {
      destroy();
    } else {
      deallocate();
      shape_ = shape;
      allocate();
    }
    construct(value);
  }

  /** Get the allocator used to allocate memory for this buffer. */
  const Alloc& get_allocator() const { return alloc_; }

  /** Get a reference to the element at `indices`. */
  reference operator() (const index_type& indices) { return base_[shape_(indices)]; }
  reference operator[] (const index_type& indices) { return base_[shape_(indices)]; }
  const_reference operator() (const index_type& indices) const { return base_[shape_(indices)]; }
  const_reference operator[] (const index_type& indices) const { return base_[shape_(indices)]; }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  reference operator() (Args... indices) { return base_[shape_(indices...)]; }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  const_reference operator() (Args... indices) const { return base_[shape_(indices...)]; }

  /** Create an `array_ref` from this array using a indices or intervals `args`.
   * Dimensions corresponding to indices in `args` are sliced, i.e. the result
   * will not have this dimension. The rest of the dimensions are cropped. */
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (const std::tuple<Args...>& args) {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator[] (const std::tuple<Args...>& args) {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (Args... args) {
    return internal::make_array_ref_at(base_, shape_, std::make_tuple(args...));
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator[] (const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator() (Args... args) const {
    return internal::make_array_ref_at(base_, shape_, std::make_tuple(args...));
  }

  /** Call a function with a reference to each value in this array. The order in
   * which `fn` is called is undefined to enable optimized memory accesses. */
  template <class Fn, class = internal::enable_if_callable<Fn, reference>>
  void for_each_value(Fn&& fn) {
    shape_traits_type::for_each_value(shape_, base_, fn);
  }
  template <class Fn, class = internal::enable_if_callable<Fn, const_reference>>
  void for_each_value(Fn&& fn) const {
    shape_traits_type::for_each_value(shape_, base_, fn);
  }

  /** Pointer to the element at the min index of the shape. */
  pointer base() { return base_; }
  const_pointer base() const { return base_; }

  /** Pointer to the element at the beginning of the flat array. */
  pointer data() { return internal::pointer_add(base_, shape_.flat_min()); }
  const_pointer data() const { return internal::pointer_add(base_, shape_.flat_min()); }

  /** Shape of this array. */
  const Shape& shape() const { return shape_; }

  template <size_t D, class = enable_if_dim<D>>
  const auto& dim() const { return shape_.template dim<D>(); }
  size_type size() const { return shape_.size(); }
  bool empty() const { return shape_.empty(); }
  bool is_compact() const { return shape_.is_compact(); }

  /** Reset the shape of this array to default. If the default constructed
   * `Shape` is empty, the array will be empty. If the default constructed
   * `Shape` is non-empty, the elements of the array will be default
   * constructed. */
  void clear() {
    deallocate();
    shape_ = Shape();
    allocate();
    construct();
  }

  /** Reallocate the array, and move the intersection of the old and new shapes
   * to the new array. */
  void reshape(Shape new_shape) {
    new_shape.resolve();
    if (shape_ == new_shape) {
      return;
    }

    // Allocate an array with the new shape.
    array new_array(new_shape);

    // Move the common elements to the new array.
    Shape intersection =
        internal::clamp(new_shape.dims(), shape_.dims(), internal::make_index_sequence<rank()>());
    pointer intersection_base =
        internal::pointer_add(new_array.base(), new_shape(intersection.min()));
    copy_shape_traits_type::for_each_value(
        shape_, base_, intersection, intersection_base, internal::move_assign<T, T>);

    *this = std::move(new_array);
  }

  /** Change the shape of the array to `new_shape`, and move the base pointer
   * by `offset`. This function is disabled for non-trivial types, because it
   * does not call the destructor or constructor for newly inaccessible or newly
   * accessible elements, respectively. */
  void set_shape(const Shape& new_shape, index_t offset = 0) {
    static_assert(std::is_trivial<value_type>::value, "set_shape is broken for non-trivial types.");
    assert(new_shape.is_resolved());
    assert(new_shape.is_subset_of(shape_, -offset));
    shape_ = new_shape;
    base_ = internal::pointer_add(base_, offset);
  }

  /** Provide some aliases for common interpretations of dimensions
   * `i`, `j`, `k` as dimensions 0, 1, 2, respectively. */
  const auto& i() const { return shape_.i(); }
  const auto& j() const { return shape_.j(); }
  const auto& k() const { return shape_.k(); }

  /** Provide some aliases for common interpretations of dimensions
   * `x`, `y`, `z` or `c`, `w` as dimensions 0, 1, 2, 3 respectively. */
  const auto& x() const { return shape_.x(); }
  const auto& y() const { return shape_.y(); }
  const auto& z() const { return shape_.z(); }
  const auto& c() const { return shape_.c(); }
  const auto& w() const { return shape_.w(); }

  /** Assuming this array represents an image with dimensions width, height,
   * channels, get the extent of those dimensions. */
  index_t width() const { return shape_.width(); }
  index_t height() const { return shape_.height(); }
  index_t channels() const { return shape_.channels(); }

  /** Assuming this array represents a matrix with dimensions {rows, cols}, get
   * the extent of those dimensions. */
  index_t rows() const { return shape_.rows(); }
  index_t columns() const { return shape_.columns(); }

  /** Compare the contents of this array to `other`. For two arrays to be
   * considered equal, they must have the same shape, and all elements
   * addressable by the shape must also be equal. */
  bool operator!=(const array& other) const { return cref() != other.cref(); }
  bool operator==(const array& other) const { return cref() == other.cref(); }

  /** Swap the contents of two arrays. This performs zero copies or moves of
   * individual elements. */
  void swap(array& other) {
    using std::swap;

    if (alloc_traits::propagate_on_container_swap::value) {
      swap(alloc_, other.alloc_);
      swap(buffer_, other.buffer_);
      swap(buffer_size_, other.buffer_size_);
      swap(base_, other.base_);
      swap(shape_, other.shape_);
    } else {
      // TODO: If the shapes are equal, we could swap each element without the
      // temporary allocation.
      array temp(std::move(other));
      other = std::move(*this);
      *this = std::move(temp);
    }
  }

  /** Make an array_ref referring to the data in this array. */
  array_ref<T, Shape> ref() { return array_ref<T, Shape>(base_, shape_); }
  const_array_ref<T, Shape> cref() const { return const_array_ref<T, Shape>(base_, shape_); }
  const_array_ref<T, Shape> ref() const { return cref(); }
  operator array_ref<T, Shape>() { return ref(); }
  operator const_array_ref<T, Shape>() const { return cref(); }
};

/** An array type with an arbitrary shape of rank `Rank`. */
template <class T, size_t Rank, class Alloc = std::allocator<T>>
using array_of_rank = array<T, shape_of_rank<Rank>, Alloc>;

/** An array type with a shape `dense_shape<Rank>`. */
template <class T, size_t Rank, class Alloc = std::allocator<T>>
using dense_array = array<T, dense_shape<Rank>, Alloc>;

/** Make a new array with shape `shape`, allocated using `alloc`. */
template <class T, class Shape, class Alloc = std::allocator<T>>
auto make_array(const Shape& shape, const Alloc& alloc = Alloc()) {
  return array<T, Shape, Alloc>(shape, alloc);
}
template <class T, class Shape, class Alloc = std::allocator<T>>
auto make_array(const Shape& shape, const T& value, const Alloc& alloc = Alloc()) {
  return array<T, Shape, Alloc>(shape, value, alloc);
}

/** Swap the contents of two arrays. */
template <class T, class Shape, class Alloc>
void swap(array<T, Shape, Alloc>& a, array<T, Shape, Alloc>& b) {
  a.swap(b);
}

/** Copy the contents of the `src` array or array_ref to the `dst` array or
 * array_ref. The interval of the shape of `dst` will be copied, and must be in
 * bounds of `src`. */
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void copy(const array_ref<TSrc, ShapeSrc>& src, const array_ref<TDst, ShapeDst>& dst) {
  if (dst.shape().empty()) {
    return;
  }
  if (!src.shape().is_in_range(dst.shape().min()) ||
      !src.shape().is_in_range(dst.shape().max())) {
    NDARRAY_THROW_OUT_OF_RANGE("dst indices out of interval of src");
  }

  copy_shape_traits<ShapeSrc, ShapeDst>::for_each_value(
      src.shape(), src.base(), dst.shape(), dst.base(), internal::copy_assign<TSrc, TDst>);
}
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst, class AllocDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void copy(const array_ref<TSrc, ShapeSrc>& src, array<TDst, ShapeDst, AllocDst>& dst) {
  copy(src, dst.ref());
}
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst, class AllocSrc,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void copy(const array<TSrc, ShapeSrc, AllocSrc>& src, const array_ref<TDst, ShapeDst>& dst) {
  copy(src.cref(), dst);
}
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst, class AllocSrc, class AllocDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void copy(const array<TSrc, ShapeSrc, AllocSrc>& src, array<TDst, ShapeDst, AllocDst>& dst) {
  copy(src.cref(), dst.ref());
}

/** Make a copy of the `src` array or array_ref with a new shape `shape`. */
template <
    class T, class ShapeSrc, class ShapeDst,
    class Alloc = std::allocator<typename std::remove_const<T>::type>,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
auto make_copy(
    const array_ref<T, ShapeSrc>& src, const ShapeDst& shape, const Alloc& alloc = Alloc()) {
  array<typename std::allocator_traits<Alloc>::value_type, ShapeDst, Alloc> dst(shape, alloc);
  copy(src, dst);
  return dst;
}
template <class T, class ShapeSrc, class ShapeDst, class AllocSrc, class AllocDst = AllocSrc,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
auto make_copy(
    const array<T, ShapeSrc, AllocSrc>& src, const ShapeDst& shape,
    const AllocDst& alloc = AllocDst()) {
  return make_copy(src.cref(), shape, alloc);
}

/** Make a copy of the `src` array or array_ref with a dense shape of the same
 * rank as `src`. */
template <
    class T, class ShapeSrc, class Alloc = std::allocator<typename std::remove_const<T>::type>>
auto make_dense_copy(const array_ref<T, ShapeSrc>& src, const Alloc& alloc = Alloc()) {
  return make_copy(src, make_dense(src.shape()), alloc);
}
template <class T, class ShapeSrc, class AllocSrc, class AllocDst = AllocSrc>
auto make_dense_copy(const array<T, ShapeSrc, AllocSrc>& src, const AllocDst& alloc = AllocDst()) {
  return make_dense_copy(src.cref(), alloc);
}

/** Make a copy of the `src` array or array_ref with a compact version of `src`s
 * shape. */
template <class T, class Shape, class Alloc = std::allocator<typename std::remove_const<T>::type>>
auto make_compact_copy(const array_ref<T, Shape>& src, const Alloc& alloc = Alloc()) {
  return make_copy(src, make_compact(src.shape()), alloc);
}
template <class T, class Shape, class AllocSrc, class AllocDst = AllocSrc>
auto make_compact_copy(const array<T, Shape, AllocSrc>& src, const AllocDst& alloc = AllocDst()) {
  return make_compact_copy(src.cref(), alloc);
}

/** Move the contents from the `src` array or array_ref to the `dst` array or
 * array_ref. The interval of the shape of `dst` will be moved, and must be in
 * bounds of `src`. */
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void move(const array_ref<TSrc, ShapeSrc>& src, const array_ref<TDst, ShapeDst>& dst) {
  if (dst.shape().empty()) {
    return;
  }
  if (!src.shape().is_in_range(dst.shape().min()) ||
      !src.shape().is_in_range(dst.shape().max())) {
    NDARRAY_THROW_OUT_OF_RANGE("dst indices out of interval of src");
  }

  copy_shape_traits<ShapeSrc, ShapeDst>::for_each_value(
      src.shape(), src.base(), dst.shape(), dst.base(), internal::move_assign<TSrc, TDst>);
}
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst, class AllocDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void move(const array_ref<TSrc, ShapeSrc>& src, array<TDst, ShapeDst, AllocDst>& dst) {
  move(src, dst.ref());
}
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst, class AllocSrc,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void move(array<TSrc, ShapeSrc, AllocSrc>& src, const array_ref<TDst, ShapeDst>& dst) {
  move(src.ref(), dst);
}
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst, class AllocSrc, class AllocDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void move(array<TSrc, ShapeSrc, AllocSrc>& src, array<TDst, ShapeDst, AllocDst>& dst) {
  move(src.ref(), dst.ref());
}
template <class T, class Shape, class Alloc>
void move(array<T, Shape, Alloc>&& src, array<T, Shape, Alloc>& dst) { dst = std::move(src); }

/** Make a copy of the `src` array or array_ref with a new shape `shape`. The
 * elements of `src` are moved to the result. */
template <class T, class ShapeSrc, class ShapeDst, class Alloc = std::allocator<T>,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
auto make_move(
    const array_ref<T, ShapeSrc>& src, const ShapeDst& shape, const Alloc& alloc = Alloc()) {
  array<typename std::allocator_traits<Alloc>::value_type, ShapeDst, Alloc> dst(shape, alloc);
  move(src, dst);
  return dst;
}
template <class T, class ShapeSrc, class ShapeDst, class AllocSrc, class AllocDst = AllocSrc,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
auto make_move(
    array<T, ShapeSrc, AllocSrc>& src, const ShapeDst& shape, const AllocDst& alloc = AllocDst()) {
  return make_move(src.ref(), shape, alloc);
}
template <class T, class Shape, class Alloc>
auto make_move(array<T, Shape, Alloc>&& src, const Shape& shape, const Alloc& alloc = Alloc()) {
  if (src.shape() == shape && alloc == src.get_allocator()) {
    return src;
  } else {
    return make_move(src.ref(), shape, alloc);
  }
}

/** Make a copy of the `src` array or array_ref with a dense shape of the same
 * rank as `src`. The elements of `src` are moved to the result. */
template <class T, class Shape, class Alloc = std::allocator<T>>
auto make_dense_move(const array_ref<T, Shape>& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_dense(src.shape()), alloc);
}
template <class T, class Shape, class AllocSrc, class AllocDst = AllocSrc>
auto make_dense_move(array<T, Shape, AllocSrc>& src, const AllocDst& alloc = AllocDst()) {
  return make_dense_move(src.ref(), alloc);
}
template <class T, class Shape, class Alloc>
auto make_dense_move(array<T, Shape, Alloc>&& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_dense(src.shape()), alloc);
}

/** Make a copy of the `src` array or array_ref with a compact version of `src`'s
 * shape. The elements of `src` are moved to the result. */
template <class T, class Shape, class Alloc = std::allocator<T>>
auto make_compact_move(const array_ref<T, Shape>& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_compact(src.shape()), alloc);
}
template <class T, class Shape, class AllocSrc, class AllocDst = AllocSrc>
auto make_compact_move(array<T, Shape, AllocSrc>& src, const AllocDst& alloc = AllocDst()) {
  return make_compact_move(src.ref(), alloc);
}
template <class T, class Shape, class Alloc>
auto make_compact_move(array<T, Shape, Alloc>&& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_compact(src.shape()), alloc);
}

/** Fill `dst` array or array_ref by copy-assigning `value`. */
template <class T, class Shape>
void fill(const array_ref<T, Shape>& dst, const T& value) {
  dst.for_each_value([value](T& i) { i = value; });
}
template <class T, class Shape, class Alloc>
void fill(array<T, Shape, Alloc>& dst, const T& value) {
  fill(dst.ref(), value);
}

/** Fill `dst` array or array_ref with the result of calling a generator
 * `g`. The order in which `g` is called is the same as
 * `shape_traits<Shape>::for_each_value`. */
template <class T, class Shape, class Generator,
    class = internal::enable_if_callable<Generator>>
void generate(const array_ref<T, Shape>& dst, Generator g) {
  dst.for_each_value([g](T& i) { i = g(); });
}
template <class T, class Shape, class Alloc, class Generator,
    class = internal::enable_if_callable<Generator>>
void generate(array<T, Shape, Alloc>& dst, Generator g) {
  generate(dst.ref(), g);
}

/** Check if two array or array_refs have equal contents. */
template <class TA, class ShapeA, class TB, class ShapeB>
bool equal(const array_ref<TA, ShapeA>& a, const array_ref<TB, ShapeB>& b) {
  if (a.shape().min() != b.shape().min() || a.shape().extent() != b.shape().extent()) {
    return false;
  }

  bool result = true;
  copy_shape_traits<ShapeA, ShapeB>::for_each_value(
      a.shape(), a.base(), b.shape(), b.base(), [&](const TA& a, const TB& b) {
    if (a != b) {
      result = false;
    }
  });
  return result;
}
template <class TA, class ShapeA, class TB, class ShapeB, class AllocB>
bool equal(const array_ref<TA, ShapeA>& a, const array<TB, ShapeB, AllocB>& b) {
  return equal(a, b.ref());
}
template <class TA, class ShapeA, class AllocA, class TB, class ShapeB>
bool equal(const array<TA, ShapeA, AllocA>& a, const array_ref<TB, ShapeB>& b) {
  return equal(a.ref(), b);
}
template <class TA, class ShapeA, class AllocA, class TB, class ShapeB, class AllocB>
bool equal(const array<TA, ShapeA, AllocA>& a, const array<TB, ShapeB, AllocB>& b) {
  return equal(a.ref(), b.ref());
}

/** Convert the shape of the array or array_ref `a` to a new type of shape
 * `NewShape`. The new shape is constructed from `a.shape()`. */
template <class NewShape, class T, class OldShape>
array_ref<T, NewShape> convert_shape(const array_ref<T, OldShape>& a) {
  return array_ref<T, NewShape>(a.base(), convert_shape<NewShape>(a.shape()));
}
template <class NewShape, class T, class OldShape, class Allocator>
array_ref<T, NewShape> convert_shape(array<T, OldShape, Allocator>& a) {
  return convert_shape<NewShape>(a.ref());
}
template <class NewShape, class T, class OldShape, class Allocator>
const_array_ref<T, NewShape> convert_shape(const array<T, OldShape, Allocator>& a) {
  return convert_shape<NewShape>(a.cref());
}

/** Reinterpret the array or array_ref `a` of type `T` to have a different type
 * `U`. The size of `T` must be equal to the size of `U`. */
template <class U, class T, class Shape,
    class = typename std::enable_if<sizeof(T) == sizeof(U)>::type>
array_ref<U, Shape> reinterpret(const array_ref<T, Shape>& a) {
  return array_ref<U, Shape>(reinterpret_cast<U*>(a.base()), a.shape());
}
template <class U, class T, class Shape, class Alloc,
    class = typename std::enable_if<sizeof(T) == sizeof(U)>::type>
array_ref<U, Shape> reinterpret(array<T, Shape, Alloc>& a) {
  return reinterpret<U>(a.ref());
}
template <class U, class T, class Shape, class Alloc,
    class = typename std::enable_if<sizeof(T) == sizeof(U)>::type>
const_array_ref<U, Shape> reinterpret(const array<T, Shape, Alloc>& a) {
  return reinterpret<const U>(a.cref());
}

/** Reinterpret the shape of the array or array_ref `a` to be a new shape
 * `new_shape`, with a base pointer offset `offset`. */
template <class NewShape, class T, class OldShape>
array_ref<T, NewShape> reinterpret_shape(
    const array_ref<T, OldShape>& a, const NewShape& new_shape, index_t offset = 0) {
  assert(new_shape.is_subset_of(a.shape(), -offset));
  return array_ref<T, NewShape>(a.base() + offset, new_shape);
}
template <class NewShape, class T, class OldShape, class Allocator>
array_ref<T, NewShape> reinterpret_shape(
    array<T, OldShape, Allocator>& a, const NewShape& new_shape, index_t offset = 0) {
  return reinterpret_shape(a.ref(), new_shape, offset);
}
template <class NewShape, class T, class OldShape, class Allocator>
const_array_ref<T, NewShape> reinterpret_shape(
    const array<T, OldShape, Allocator>& a, const NewShape& new_shape, index_t offset = 0) {
  return reinterpret_shape(a.cref(), new_shape, offset);
}

/** Reinterpret the shape of the array or array_ref `a` to be transposed
 * or reordered using `transpose<DimIndices...>(a.shape())` or
 * `reorder<DimIndices...>(a.shape())`. */
template <size_t... DimIndices, class T, class OldShape,
    class = internal::enable_if_permutation<OldShape::rank(), DimIndices...>>
auto transpose(const array_ref<T, OldShape>& a) {
  return reinterpret_shape(a, transpose<DimIndices...>(a.shape()));
}
template <size_t... DimIndices, class T, class OldShape, class Allocator,
    class = internal::enable_if_permutation<OldShape::rank(), DimIndices...>>
auto transpose(array<T, OldShape, Allocator>& a) {
  return reinterpret_shape(a, transpose<DimIndices...>(a.shape()));
}
template <size_t... DimIndices, class T, class OldShape, class Allocator,
    class = internal::enable_if_permutation<OldShape::rank(), DimIndices...>>
auto transpose(const array<T, OldShape, Allocator>& a) {
  return reinterpret_shape(a, transpose<DimIndices...>(a.shape()));
}
template <size_t... DimIndices, class T, class OldShape>
auto reorder(const array_ref<T, OldShape>& a) {
  return reinterpret_shape(a, reorder<DimIndices...>(a.shape()));
}
template <size_t... DimIndices, class T, class OldShape, class Allocator>
auto reorder(array<T, OldShape, Allocator>& a) {
  return reinterpret_shape(a, reorder<DimIndices...>(a.shape()));
}
template <size_t... DimIndices, class T, class OldShape, class Allocator>
auto reorder(const array<T, OldShape, Allocator>& a) {
  return reinterpret_shape(a, reorder<DimIndices...>(a.shape()));
}

/** Allocator satisfying the `std::allocator` interface which allocates memory
 * from a buffer with automatic storage. This can only be used with containers
 * that have a maximum of one concurrent live allocation, which is the case for
 * `array`. */
template <class T, size_t N, size_t Alignment = sizeof(T)>
class auto_allocator {
  alignas(Alignment) char buffer[N * sizeof(T)];
  bool allocated;

 public:
  using value_type = T;

  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::false_type;

  static auto_allocator select_on_container_copy_construction(const auto_allocator&) {
    return auto_allocator();
  }

  auto_allocator() : allocated(false) {}
  template <class U, size_t U_N> constexpr
  auto_allocator(const auto_allocator<U, U_N>&) noexcept : allocated(false) {}
  // These constructors/assignment operators are workarounds for a C++
  // STL implementation not respecting the propagate typedefs or the
  // 'select_on_...' function. (https://github.com/dsharlet/array/issues/7)
  auto_allocator(const auto_allocator&) noexcept : allocated(false) {}
  auto_allocator(auto_allocator&&) noexcept : allocated(false) {}
  auto_allocator& operator=(const auto_allocator&) { return *this; }
  auto_allocator& operator=(auto_allocator&&) { return *this; }

  value_type* allocate(size_t n) {
    if (allocated) NDARRAY_THROW_BAD_ALLOC();
    if (n > N) NDARRAY_THROW_BAD_ALLOC();
    allocated = true;
    return reinterpret_cast<value_type*>(&buffer[0]);
  }
  void deallocate(value_type*, size_t) noexcept {
    allocated = false;
  }

  template <class U, size_t U_N>
  friend bool operator==(const auto_allocator<T, N>& a, const auto_allocator<U, U_N>& b) {
    return &a.buffer[0] == &b.buffer[0];
  }

  template <class U, size_t U_N>
  friend bool operator!=(const auto_allocator<T, N>& a, const auto_allocator<U, U_N>& b) {
    return &a.buffer[0] != &b.buffer[0];
  }
};

/** Allocator satisfying the `std::allocator` interface that is a wrapper
 * around another allocator `BaseAlloc`, and skips default construction.
 * Using this allocator can be dangerous. It is only safe to use when
 * `BaseAlloc::value_type` is a trivial type. */
template <class BaseAlloc>
class uninitialized_allocator : public BaseAlloc {
 public:
  using value_type =
      typename std::allocator_traits<BaseAlloc>::value_type;

  using propagate_on_container_copy_assignment =
      typename std::allocator_traits<BaseAlloc>::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename std::allocator_traits<BaseAlloc>::propagate_on_container_move_assignment;
  using propagate_on_container_swap =
      typename std::allocator_traits<BaseAlloc>::propagate_on_container_swap;
  static uninitialized_allocator select_on_container_copy_construction(
      const uninitialized_allocator& alloc) {
    return std::allocator_traits<BaseAlloc>::select_on_container_copy_construction(alloc);
  }

  value_type* allocate(size_t n) {
    return std::allocator_traits<BaseAlloc>::allocate(*this, n);
  }
  void deallocate(value_type* p, size_t n) noexcept {
    return std::allocator_traits<BaseAlloc>::deallocate(*this, p, n);
  }

  // TODO: Consider adding an enable_if to this to disable it for
  // non-trivial value_types.
  template <class... Args>
  NDARRAY_INLINE void construct(value_type* ptr, Args&&... args) {
    // Skip default construction.
    if (sizeof...(Args) > 0) {
      std::allocator_traits<BaseAlloc>::construct(*this, ptr, std::forward<Args>(args)...);
    }
  }
};

/** Allocator equivalent to `std::allocator<T>` that does not default
 * construct values. */
template <class T, class = typename std::enable_if<std::is_trivial<T>::value>::type>
using uninitialized_std_allocator = uninitialized_allocator<std::allocator<T>>;

/** Allocator equivalent to `auto_allocator<T, N, Alignment>` that
 * does not default construct values. */
template <class T, size_t N, size_t Alignment = sizeof(T),
    class = typename std::enable_if<std::is_trivial<T>::value>::type>
using uninitialized_auto_allocator = uninitialized_allocator<auto_allocator<T, N, Alignment>>;

}  // namespace nda

#endif  // NDARRAY_ARRAY_H

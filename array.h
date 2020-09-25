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
// TODO(jiawen): CUDA *should* support assert on device. This might be due to the fact that we are
// not depending on the CUDA toolkit.
#if defined(__CUDA__)
#undef assert
#define assert(e)
#else
#include <cassert>
#endif

#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

// Some things in this header are unbearably slow without optimization if they
// don't get inlined.
#if defined(__GNUC__)
#define NDARRAY_INLINE inline __attribute__((always_inline))
#elif defined(__clang__)
#if defined(__CUDA__)
#define NDARRAY_INLINE __forceinline__
#else
#define NDARRAY_INLINE inline __attribute__((always_inline))
#endif
#else
#define NDARRAY_INLINE inline
#endif

// Many of the functions in this header are templates that are usually unique
// specializations, which are beneficial to inline. The compiler will inline
// functions it knows are used only once, but it can't know this unless the
// functions have internal linkage.
#define NDARRAY_UNIQUE static

// Functions attributed with NDARRAY_HOST_DEVICE can run on both device and host mode on CUDA.
#if defined(__CUDA__)
#define NDARRAY_HOST_DEVICE __device__ __host__
#else
#define NDARRAY_HOST_DEVICE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define NDARRAY_RESTRICT __restrict__
#else
#define NDARRAY_RESTRICT
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

// Workaround CUDA not supporting std::declval.
// https://stackoverflow.com/questions/31969644/compilation-error-with-nvcc-and-c11-need-minimal-failing-example
template <typename T>
NDARRAY_HOST_DEVICE typename std::add_rvalue_reference<T>::type declval() noexcept;

NDARRAY_INLINE constexpr index_t abs(index_t x) { return x >= 0 ? x : -x; }

NDARRAY_INLINE constexpr index_t is_static(index_t x) { return x != dynamic; }
NDARRAY_INLINE constexpr index_t is_dynamic(index_t x) { return x == dynamic; }

constexpr bool is_dynamic(index_t a, index_t b) { return is_dynamic(a) || is_dynamic(b); }

// Returns true if a and b are statically not equal, but they may still be
// dynamically not equal even if this returns false.
constexpr bool not_equal(index_t a, index_t b) { return is_static(a) && is_static(b) && a != b; }

template <index_t A, index_t B>
using disable_if_not_equal = std::enable_if_t<!not_equal(A, B)>;

// Math for (possibly) static values.
constexpr index_t static_abs(index_t x) { return is_dynamic(x) ? dynamic : abs(x); }
constexpr index_t static_add(index_t a, index_t b) { return is_dynamic(a, b) ? dynamic : a + b; }
constexpr index_t static_sub(index_t a, index_t b) { return is_dynamic(a, b) ? dynamic : a - b; }
constexpr index_t static_mul(index_t a, index_t b) { return is_dynamic(a, b) ? dynamic : a * b; }
constexpr index_t static_min(index_t a, index_t b) {
  return is_dynamic(a, b) ? dynamic : (a < b ? a : b);
}
constexpr index_t static_max(index_t a, index_t b) {
  return is_dynamic(a, b) ? dynamic : (a > b ? a : b);
}

// A type that mimics a constexpr index_t with value Value, unless Value is
// dynamic, then mimics index_t.
template <index_t Value>
struct constexpr_index {
public:
  // These asserts are really hard to debug
  // https://github.com/dsharlet/array/issues/26
  NDARRAY_HOST_DEVICE constexpr_index(index_t value = Value) { assert(value == Value); }
  NDARRAY_HOST_DEVICE constexpr_index& operator=(index_t value) {
    assert(value == Value);
    return *this;
  }
  NDARRAY_HOST_DEVICE NDARRAY_INLINE operator index_t() const { return Value; }
};

template <>
struct constexpr_index<dynamic> {
  index_t value_;

public:
  NDARRAY_HOST_DEVICE constexpr_index(index_t value) : value_(value) {}
  NDARRAY_HOST_DEVICE constexpr_index& operator=(index_t value) {
    value_ = value;
    return *this;
  }
  NDARRAY_INLINE NDARRAY_HOST_DEVICE operator index_t() const { return value_; }
};

} // namespace internal

/** An iterator representing an index. */
class index_iterator {
  index_t i_;

public:
  /** Construct the iterator with an index `i`. */
  index_iterator(index_t i) : i_(i) {}

  /** Access the current index of this iterator. */
  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_t operator*() const { return i_; }

  NDARRAY_INLINE NDARRAY_HOST_DEVICE bool operator==(const index_iterator& r) const {
    return i_ == r.i_;
  }
  NDARRAY_INLINE NDARRAY_HOST_DEVICE bool operator!=(const index_iterator& r) const {
    return i_ != r.i_;
  }

  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_iterator operator++(int) { return index_iterator(i_++); }
  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_iterator& operator++() {
    ++i_;
    return *this;
  }
};

template <index_t Min, index_t Extent, index_t Stride>
class dim;

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
  internal::constexpr_index<Min_> min_;
  internal::constexpr_index<Extent_> extent_;

public:
  static constexpr index_t Min = Min_;
  static constexpr index_t Extent = Extent_;
  static constexpr index_t Max = internal::static_sub(internal::static_add(Min, Extent), 1);

  /** Construct a new interval object. If the `min` or `extent` is specified in
   * the constructor, it must have a value compatible with `Min` or `Extent`,
   * respectively.
   *
   * The default values if not specified in the constructor are:
   * - The default `min` is `Min` if `Min` is static, or 0 if not.
   * - The default `extent` is `Extent` if `Extent` is static, or 1 if not. */
  NDARRAY_HOST_DEVICE interval(index_t min, index_t extent) : min_(min), extent_(extent) {}
  NDARRAY_HOST_DEVICE interval(index_t min)
      : interval(min, internal::is_static(Extent) ? Extent : 1) {}
  NDARRAY_HOST_DEVICE interval() : interval(internal::is_static(Min) ? Min : 0) {}

  NDARRAY_HOST_DEVICE interval(const interval&) = default;
  NDARRAY_HOST_DEVICE interval(interval&&) = default;
  NDARRAY_HOST_DEVICE interval& operator=(const interval&) = default;
  NDARRAY_HOST_DEVICE interval& operator=(interval&&) = default;

  /** Copy construction or assignment of another interval object, possibly
   * with different compile-time template parameters. `other.min()` and
   * `other.extent()` must be compatible with `Min` and `Extent`, respectively. */
  template <index_t CopyMin, index_t CopyExtent,
      class = internal::disable_if_not_equal<Min, CopyMin>,
      class = internal::disable_if_not_equal<Extent, CopyExtent>>
  NDARRAY_HOST_DEVICE interval(const interval<CopyMin, CopyExtent>& other)
      : interval(other.min(), other.extent()) {}
  template <index_t CopyMin, index_t CopyExtent,
      class = internal::disable_if_not_equal<Min, CopyMin>,
      class = internal::disable_if_not_equal<Extent, CopyExtent>>
  NDARRAY_HOST_DEVICE interval& operator=(const interval<CopyMin, CopyExtent>& other) {
    set_min(other.min());
    set_extent(other.extent());
    return *this;
  }

  /** Get or set the first index in this interval. */
  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_t min() const { return min_; }
  NDARRAY_INLINE NDARRAY_HOST_DEVICE void set_min(index_t min) { min_ = min; }
  /** Get or set the number of indices in this interval. */
  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_t extent() const { return extent_; }
  NDARRAY_INLINE NDARRAY_HOST_DEVICE void set_extent(index_t extent) { extent_ = extent; }

  /** Get or set the last index in this interval. */
  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_t max() const { return min_ + extent_ - 1; }
  NDARRAY_INLINE NDARRAY_HOST_DEVICE void set_max(index_t max) { set_extent(max - min_ + 1); }

  /** Returns true if `at` is within the interval `[min(), max()]`. */
  NDARRAY_INLINE NDARRAY_HOST_DEVICE bool is_in_range(index_t at) const {
    return min_ <= at && at <= max();
  }
  /** Returns true if `at.min()` and `at.max()` are both within the interval
   * `[min(), max()]`. */
  template <index_t OtherMin, index_t OtherExtent>
  NDARRAY_INLINE NDARRAY_HOST_DEVICE bool is_in_range(
      const interval<OtherMin, OtherExtent>& at) const {
    return min_ <= at.min() && at.max() <= max();
  }
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride>
  NDARRAY_INLINE NDARRAY_HOST_DEVICE bool is_in_range(
      const dim<OtherMin, OtherExtent, OtherStride>& at) const {
    return min_ <= at.min() && at.max() <= max();
  }

  /** Make an iterator referring to the first index in this interval. */
  NDARRAY_HOST_DEVICE index_iterator begin() const { return index_iterator(min_); }
  /** Make an iterator referring to one past the last index in this interval. */
  NDARRAY_HOST_DEVICE index_iterator end() const { return index_iterator(max() + 1); }

  /** Two interval objects are considered equal if they contain the
   * same indices. */
  template <index_t OtherMin, index_t OtherExtent>
  NDARRAY_HOST_DEVICE bool operator==(const interval<OtherMin, OtherExtent>& other) const {
    return min_ == other.min() && extent_ == other.extent();
  }
  template <index_t OtherMin, index_t OtherExtent>
  NDARRAY_HOST_DEVICE bool operator!=(const interval<OtherMin, OtherExtent>& other) const {
    return !operator==(other);
  }
};

/** An alias of `interval` with a fixed extent and dynamic min. */
template <index_t Extent>
using fixed_interval = interval<dynamic, Extent>;

/** Make an interval from a half-open range `[begin, end)`. */
NDARRAY_INLINE NDARRAY_HOST_DEVICE interval<> range(index_t begin, index_t end) {
  return interval<>(begin, end - begin);
}
NDARRAY_INLINE NDARRAY_HOST_DEVICE interval<> r(index_t begin, index_t end) {
  return interval<>(begin, end - begin);
}

/** Make an interval from a half-open range `[begin, begin + Extent)`. */
template <index_t Extent>
NDARRAY_HOST_DEVICE fixed_interval<Extent> range(index_t begin) {
  return fixed_interval<Extent>(begin);
}
template <index_t Extent>
NDARRAY_HOST_DEVICE fixed_interval<Extent> r(index_t begin) {
  return fixed_interval<Extent>(begin);
}

/** Placeholder object representing an interval that indicates keep
 * the whole dimension when used in an indexing expression. */
const interval<0, -1> all, _;

/** Overloads of `std::begin` and `std::end` for an interval. */
template <index_t Min, index_t Extent>
NDARRAY_HOST_DEVICE index_iterator begin(const interval<Min, Extent>& d) {
  return d.begin();
}
template <index_t Min, index_t Extent>
NDARRAY_HOST_DEVICE index_iterator end(const interval<Min, Extent>& d) {
  return d.end();
}

/** Clamp `x` to the interval [min, max]. */
NDARRAY_INLINE NDARRAY_HOST_DEVICE index_t clamp(index_t x, index_t min, index_t max) {
  return std::min(std::max(x, min), max);
}

/** Clamp `x` to the range described by an object `r` with a `min()` and
 * `max()` method. */
template <class Range>
NDARRAY_HOST_DEVICE index_t clamp(index_t x, const Range& r) {
  return clamp(x, r.min(), r.max());
}

/** Describes one dimension of an array. The template parameters enable
 * providing compile time constants for the `min`, `extent`, and `stride` of the
 * dim.
 *
 * These parameters define a mapping from the indices of the dimension to
 * offsets: `offset(x) = (x - min)*stride`. The extent does not affect the
 * mapping directly. Values not in the interval `[min, min + extent)` are considered
 * to be out of bounds. */
template <index_t Min_ = dynamic, index_t Extent_ = dynamic, index_t Stride_ = dynamic>
class dim : protected interval<Min_, Extent_> {
public:
  using base_range = interval<Min_, Extent_>;

protected:
  internal::constexpr_index<Stride_> stride_;

  using base_range::extent_;
  using base_range::min_;

public:
  using base_range::Extent;
  using base_range::Max;
  using base_range::Min;

  static constexpr index_t Stride = Stride_;

  /** Construct a new dim object. If the `min`, `extent` or `stride` are
   * specified in the constructor, it must have a value compatible with `Min`,
   * `Extent`, or `Stride`, respectively.
   *
   * The default values if not specified in the constructor are:
   * - The default `min` is `Min` if `Min` is static, or 0 if not.
   * - The default `extent` is `Extent` if `Extent` is static, or 0 if not.
   * - The default `stride` is `Stride`. */
  NDARRAY_HOST_DEVICE dim(index_t min, index_t extent, index_t stride = Stride)
      : base_range(min, extent), stride_(stride) {}
  NDARRAY_HOST_DEVICE dim(index_t extent) : dim(internal::is_static(Min) ? Min : 0, extent) {}
  NDARRAY_HOST_DEVICE dim() : dim(internal::is_static(Extent) ? Extent : 0) {}

  NDARRAY_HOST_DEVICE dim(const base_range& interval, index_t stride = Stride)
      : dim(interval.min(), interval.extent(), stride) {}
  NDARRAY_HOST_DEVICE dim(const dim&) = default;
  NDARRAY_HOST_DEVICE dim(dim&&) = default;
  NDARRAY_HOST_DEVICE dim& operator=(const dim&) = default;
  NDARRAY_HOST_DEVICE dim& operator=(dim&&) = default;

  /** Copy construction or assignment of another dim object, possibly
   * with different compile-time template parameters. `other.min()`,
   * `other.extent()`, and `other.stride()` must be compatible with `Min`,
   * `Extent`, and `Stride`, respectively. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride,
      class = internal::disable_if_not_equal<Min, CopyMin>,
      class = internal::disable_if_not_equal<Extent, CopyExtent>,
      class = internal::disable_if_not_equal<Stride, CopyStride>>
  NDARRAY_HOST_DEVICE dim(const dim<CopyMin, CopyExtent, CopyStride>& other)
      : dim(other.min(), other.extent(), other.stride()) {}
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride,
      class = internal::disable_if_not_equal<Min, CopyMin>,
      class = internal::disable_if_not_equal<Extent, CopyExtent>,
      class = internal::disable_if_not_equal<Stride, CopyStride>>
  NDARRAY_HOST_DEVICE dim& operator=(const dim<CopyMin, CopyExtent, CopyStride>& other) {
    set_min(other.min());
    set_extent(other.extent());
    set_stride(other.stride());
    return *this;
  }

  using base_range::begin;
  using base_range::end;
  using base_range::extent;
  using base_range::is_in_range;
  using base_range::max;
  using base_range::min;
  using base_range::set_extent;
  using base_range::set_max;
  using base_range::set_min;

  /** Get or set the distance in flat indices between neighboring elements
   * in this dim. */
  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_t stride() const { return stride_; }
  NDARRAY_INLINE NDARRAY_HOST_DEVICE void set_stride(index_t stride) { stride_ = stride; }

  /** Offset of the index `at` in this dim in the flat array. */
  NDARRAY_INLINE NDARRAY_HOST_DEVICE index_t flat_offset(index_t at) const {
    return (at - min_) * stride_;
  }

  /** Two dim objects are considered equal if their mins, extents, and strides
   * are equal. */
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride>
  NDARRAY_HOST_DEVICE bool operator==(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
    return min_ == other.min() && extent_ == other.extent() && stride_ == other.stride();
  }
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride>
  NDARRAY_HOST_DEVICE bool operator!=(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
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

// An iterator for a range of intervals.
template <index_t InnerExtent = dynamic>
class split_iterator {
  fixed_interval<InnerExtent> i;
  index_t outer_max;

public:
  NDARRAY_HOST_DEVICE split_iterator(const fixed_interval<InnerExtent>& i, index_t outer_max)
      : i(i), outer_max(outer_max) {}

  NDARRAY_HOST_DEVICE bool operator==(const split_iterator& r) const {
    return i.min() == r.i.min();
  }
  NDARRAY_HOST_DEVICE bool operator!=(const split_iterator& r) const {
    return i.min() != r.i.min();
  }

  NDARRAY_HOST_DEVICE fixed_interval<InnerExtent> operator*() const { return i; }

  NDARRAY_HOST_DEVICE split_iterator& operator++() {
    if (is_static(InnerExtent)) {
      // When the extent of the inner split is a compile-time constant,
      // we can't shrink the out of bounds interval. Instead, shift the min,
      // assuming the outer dimension is bigger than the inner extent.
      i.set_min(i.min() + InnerExtent);
      // Only shift the min when this straddles the end of the buffer,
      // so the iterator can advance to the end (one past the max).
      if (i.min() <= outer_max && i.max() > outer_max) { i.set_min(outer_max - InnerExtent + 1); }
    } else {
      // When the extent of the inner split is not a compile-time constant,
      // we can just modify the extent.
      i.set_min(i.min() + i.extent());
      index_t max = std::min(i.max(), outer_max);
      i.set_extent(max - i.min() + 1);
    }
    return *this;
  }
  NDARRAY_HOST_DEVICE split_iterator operator++(int) {
    split_iterator<InnerExtent> result(*this);
    ++*this;
    return result;
  }
};

// TODO: Remove this when std::iterator_range is standard.
template <class T>
class iterator_range {
  T begin_;
  T end_;

public:
  NDARRAY_HOST_DEVICE iterator_range(T begin, T end) : begin_(begin), end_(end) {}

  NDARRAY_HOST_DEVICE T begin() const { return begin_; }
  NDARRAY_HOST_DEVICE T end() const { return end_; }
};

template <index_t InnerExtent = dynamic>
using split_iterator_range = iterator_range<split_iterator<InnerExtent>>;

} // namespace internal

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
NDARRAY_HOST_DEVICE internal::split_iterator_range<InnerExtent> split(
    const interval<Min, Extent>& v) {
  assert(v.extent() >= InnerExtent);
  return {{fixed_interval<InnerExtent>(v.min()), v.max()},
      {fixed_interval<InnerExtent>(v.max() + 1), v.max()}};
}
template <index_t InnerExtent, index_t Min, index_t Extent, index_t Stride>
NDARRAY_HOST_DEVICE internal::split_iterator_range<InnerExtent> split(
    const dim<Min, Extent, Stride>& v) {
  return split<InnerExtent>(interval<Min, Extent>(v.min(), v.extent()));
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
NDARRAY_HOST_DEVICE internal::split_iterator_range<> split(
    const interval<Min, Extent>& v, index_t inner_extent) {
  return {{interval<>(v.min(), std::min(inner_extent, v.extent())), v.max()},
      {interval<>(v.max() + 1, 0), v.max()}};
}
template <index_t Min, index_t Extent, index_t Stride>
NDARRAY_HOST_DEVICE internal::split_iterator_range<> split(
    const dim<Min, Extent, Stride>& v, index_t inner_extent) {
  return split(interval<Min, Extent>(v.min(), v.extent()), inner_extent);
}

namespace internal {

using std::index_sequence;
using std::make_index_sequence;

// Call `fn` with the elements of tuple `args` unwrapped from the tuple.
// TODO: When we assume C++17, this can be replaced by std::apply.
template <class Fn, class Args, size_t... Is>
NDARRAY_INLINE NDARRAY_HOST_DEVICE auto apply(Fn&& fn, const Args& args, index_sequence<Is...>)
    -> decltype(fn(std::get<Is>(args)...)) {
  return fn(std::get<Is>(args)...);
}
template <class Fn, class... Args>
NDARRAY_INLINE NDARRAY_HOST_DEVICE auto apply(Fn&& fn, const std::tuple<Args...>& args)
    -> decltype(internal::apply(fn, args, make_index_sequence<sizeof...(Args)>())) {
  return internal::apply(fn, args, make_index_sequence<sizeof...(Args)>());
}

template <class Fn, class... Args>
using enable_if_callable = decltype(internal::declval<Fn>()(internal::declval<Args>()...));
template <class Fn, class Args>
using enable_if_applicable =
    decltype(internal::apply(internal::declval<Fn>(), internal::declval<Args>()));

// Some variadic reduction helpers.
NDARRAY_INLINE constexpr index_t sum() { return 0; }
NDARRAY_INLINE constexpr index_t sum(index_t x0) { return x0; }
NDARRAY_INLINE constexpr index_t sum(index_t x0, index_t x1) { return x0 + x1; }
NDARRAY_INLINE constexpr index_t sum(index_t x0, index_t x1, index_t x2) { return x0 + x1 + x2; }
template <class... Rest>
NDARRAY_INLINE constexpr index_t sum(index_t x0, index_t x1, index_t x2, index_t x3, Rest... rest) {
  return x0 + x1 + x2 + x3 + sum(rest...);
}

NDARRAY_INLINE constexpr int product() { return 1; }
template <class T, class... Rest>
NDARRAY_INLINE constexpr T product(T first, Rest... rest) {
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
NDARRAY_HOST_DEVICE index_t product(const Tuple& t, index_sequence<Is...>) {
  return product(std::get<Is>(t)...);
}

// Returns true if all of bools are true.
template <class... Bools>
constexpr bool all(Bools... bools) {
  return sum((bools ? 0 : 1)...) == 0;
}
template <class... Bools>
constexpr bool any(Bools... bools) {
  return sum((bools ? 1 : 0)...) != 0;
}

// Computes the sum of the offsets of a list of dims and indices.
template <class Dims, class Indices, size_t... Is>
NDARRAY_HOST_DEVICE index_t flat_offset(
    const Dims& dims, const Indices& indices, index_sequence<Is...>) {
  return sum(std::get<Is>(dims).flat_offset(std::get<Is>(indices))...);
}

// Computes one more than the sum of the offsets of the last index in every dim.
template <class Dims, size_t... Is>
NDARRAY_HOST_DEVICE index_t flat_min(const Dims& dims, index_sequence<Is...>) {
  return sum(
      (std::get<Is>(dims).extent() - 1) * std::min<index_t>(0, std::get<Is>(dims).stride())...);
}

template <class Dims, size_t... Is>
NDARRAY_HOST_DEVICE index_t flat_max(const Dims& dims, index_sequence<Is...>) {
  return sum(
      (std::get<Is>(dims).extent() - 1) * std::max<index_t>(0, std::get<Is>(dims).stride())...);
}

// Make dims with the interval of the first parameter and the stride
// of the second parameter.
template <index_t DimMin, index_t DimExtent, index_t DimStride>
NDARRAY_HOST_DEVICE auto range_with_stride(index_t x, const dim<DimMin, DimExtent, DimStride>& d) {
  return dim<dynamic, 1, DimStride>(x, 1, d.stride());
}
template <index_t CropMin, index_t CropExtent, index_t DimMin, index_t DimExtent, index_t Stride>
NDARRAY_HOST_DEVICE auto range_with_stride(
    const interval<CropMin, CropExtent>& x, const dim<DimMin, DimExtent, Stride>& d) {
  return dim<CropMin, CropExtent, Stride>(x.min(), x.extent(), d.stride());
}
template <index_t CropMin, index_t CropExtent, index_t CropStride, index_t DimMin,
    index_t DimExtent, index_t Stride>
NDARRAY_HOST_DEVICE auto range_with_stride(
    const dim<CropMin, CropExtent, CropStride>& x, const dim<DimMin, DimExtent, Stride>& d) {
  return dim<CropMin, CropExtent, Stride>(x.min(), x.extent(), d.stride());
}
template <index_t Min, index_t Extent, index_t Stride>
NDARRAY_HOST_DEVICE auto range_with_stride(const decltype(_)&, const dim<Min, Extent, Stride>& d) {
  return d;
}

template <class Intervals, class Dims, size_t... Is>
NDARRAY_HOST_DEVICE auto intervals_with_strides(
    const Intervals& intervals, const Dims& dims, index_sequence<Is...>) {
  return std::make_tuple(range_with_stride(std::get<Is>(intervals), std::get<Is>(dims))...);
}

// Make a tuple of dims corresponding to elements in intervals that are not slices.
template <class Dim>
NDARRAY_HOST_DEVICE std::tuple<> skip_slices_impl(const Dim& d, index_t) {
  return std::tuple<>();
}
template <class Dim, index_t Min, index_t Extent>
NDARRAY_HOST_DEVICE std::tuple<Dim> skip_slices_impl(const Dim& d, const interval<Min, Extent>&) {
  return std::tuple<Dim>(d);
}
template <class Dim, index_t Min, index_t Extent, index_t Stride>
NDARRAY_HOST_DEVICE std::tuple<Dim> skip_slices_impl(
    const Dim& d, const dim<Min, Extent, Stride>&) {
  return std::tuple<Dim>(d);
}

template <class Dims, class Intervals, size_t... Is>
NDARRAY_HOST_DEVICE auto skip_slices(
    const Dims& dims, const Intervals& intervals, index_sequence<Is...>) {
  return std::tuple_cat(skip_slices_impl(std::get<Is>(dims), std::get<Is>(intervals))...);
}

// Checks if all indices are in interval of each corresponding dim.
template <class Dims, class Indices, size_t... Is>
NDARRAY_HOST_DEVICE bool is_in_range(
    const Dims& dims, const Indices& indices, index_sequence<Is...>) {
  return all(std::get<Is>(dims).is_in_range(std::get<Is>(indices))...);
}

// Get the mins of a series of intervals.
template <class Dim>
NDARRAY_HOST_DEVICE index_t min_of_range(index_t x, const Dim&) {
  return x;
}
template <index_t Min, index_t Extent, class Dim>
NDARRAY_HOST_DEVICE index_t min_of_range(const interval<Min, Extent>& x, const Dim&) {
  return x.min();
}
template <index_t Min, index_t Extent, index_t Stride, class Dim>
NDARRAY_HOST_DEVICE index_t min_of_range(const dim<Min, Extent, Stride>& x, const Dim&) {
  return x.min();
}
template <class Dim>
NDARRAY_HOST_DEVICE index_t min_of_range(const decltype(_)&, const Dim& dim) {
  return dim.min();
}

template <class Intervals, class Dims, size_t... Is>
NDARRAY_HOST_DEVICE auto mins_of_intervals(
    const Intervals& intervals, const Dims& dims, index_sequence<Is...>) {
  return std::make_tuple(min_of_range(std::get<Is>(intervals), std::get<Is>(dims))...);
}

template <class... Dims, size_t... Is>
NDARRAY_HOST_DEVICE auto mins(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).min()...);
}

template <class... Dims, size_t... Is>
NDARRAY_HOST_DEVICE auto extents(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).extent()...);
}

template <class... Dims, size_t... Is>
NDARRAY_HOST_DEVICE auto strides(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).stride()...);
}

template <class... Dims, size_t... Is>
NDARRAY_HOST_DEVICE auto maxs(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).max()...);
}

// The following series of functions implements the algorithm for
// automatically determining what unknown dynamic strides should be.

// A proposed stride is "OK" w.r.t. `dim` if the proposed
// stride does not intersect the dim.
template <class Dim>
NDARRAY_HOST_DEVICE bool is_stride_ok(index_t stride, index_t extent, const Dim& dim) {
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

// Replace strides that are not OK with values that cannot be the
// smallest stride.
template <class... Dims>
NDARRAY_HOST_DEVICE index_t filter_stride(index_t stride, index_t extent, const Dims&... dims) {
  if (all(is_stride_ok(stride, extent, dims)...)) {
    return stride;
  } else {
    return std::numeric_limits<index_t>::max();
  }
}

// The candidate stride for some other dimension is the minimum stride it
// could have without intersecting this dim.
template <class Dim>
NDARRAY_HOST_DEVICE index_t candidate_stride(const Dim& dim) {
  if (is_dynamic(dim.stride())) {
    return std::numeric_limits<index_t>::max();
  } else {
    return std::max<index_t>(1, abs(dim.stride()) * dim.extent());
  }
}

// Find the best stride (the smallest) out of all possible candidate strides.
template <class Dims, size_t... Is>
NDARRAY_HOST_DEVICE index_t find_stride(index_t extent, const Dims& dims, index_sequence<Is...>) {
  return variadic_min(filter_stride(1, extent, std::get<Is>(dims)...),
      filter_stride(candidate_stride(std::get<Is>(dims)), extent, std::get<Is>(dims)...)...);
}

// Replace unknown dynamic strides for each dimension, starting with the first dimension.
template <class AllDims>
NDARRAY_HOST_DEVICE void resolve_unknown_strides(AllDims& all_dims) {}
template <class AllDims, class Dim0, class... Dims>
NDARRAY_HOST_DEVICE void resolve_unknown_strides(AllDims& all_dims, Dim0& dim0, Dims&... dims) {
  if (is_dynamic(dim0.stride())) {
    constexpr size_t rank = std::tuple_size<AllDims>::value;
    dim0.set_stride(find_stride(dim0.extent(), all_dims, make_index_sequence<rank>()));
  }
  resolve_unknown_strides(all_dims, dims...);
}

template <class Dims, size_t... Is>
NDARRAY_HOST_DEVICE void resolve_unknown_strides(Dims& dims, index_sequence<Is...>) {
  resolve_unknown_strides(dims, std::get<Is>(dims)...);
}

template <class Dims, size_t... Is>
NDARRAY_HOST_DEVICE bool is_resolved(const Dims& dims, index_sequence<Is...>) {
  return all(!is_dynamic(std::get<Is>(dims).stride())...);
}

// A helper to transform an array to a tuple.
template <class T, class Tuple, size_t... Is>
NDARRAY_HOST_DEVICE std::array<T, sizeof...(Is)> tuple_to_array(
    const Tuple& t, index_sequence<Is...>) {
  return {{std::get<Is>(t)...}};
}

template <class T, class... Ts>
NDARRAY_HOST_DEVICE std::array<T, sizeof...(Ts)> tuple_to_array(const std::tuple<Ts...>& t) {
  return tuple_to_array<T>(t, make_index_sequence<sizeof...(Ts)>());
}

template <class T, size_t N, size_t... Is>
NDARRAY_HOST_DEVICE auto array_to_tuple(const std::array<T, N>& a, index_sequence<Is...>) {
  return std::make_tuple(a[Is]...);
}
template <class T, size_t N>
NDARRAY_HOST_DEVICE auto array_to_tuple(const std::array<T, N>& a) {
  return array_to_tuple(a, make_index_sequence<N>());
}

template <class T, size_t N>
using tuple_of_n = decltype(array_to_tuple(internal::declval<std::array<T, N>>()));

// A helper to check if a parameter pack is entirely implicitly convertible to
// any type Ts, for use with std::enable_if
template <class T, class... Args>
struct all_of_any_type : std::false_type {};
template <class T>
struct all_of_any_type<T> : std::true_type {};
template <class... Ts, class Arg, class... Args>
struct all_of_any_type<std::tuple<Ts...>, Arg, Args...> {
  static constexpr bool value = any(std::is_constructible<Ts, Arg>::value...) &&
                                all_of_any_type<std::tuple<Ts...>, Args...>::value;
};

// Wrapper for checking if a parameter pack is entirely implicitly convertible
// to one type T.
template <class T, class... Args>
using all_of_type = all_of_any_type<std::tuple<T>, Args...>;

template <size_t I, class T, class... Us, std::enable_if_t<(I < sizeof...(Us)), int> = 0>
NDARRAY_HOST_DEVICE auto convert_dim(const std::tuple<Us...>& u) {
  return std::get<I>(u);
}
template <size_t I, class T, class... Us, std::enable_if_t<(I >= sizeof...(Us)), int> = 0>
NDARRAY_HOST_DEVICE auto convert_dim(const std::tuple<Us...>& u) {
  // For dims beyond the rank of U, make a dimension of type T_I with extent 1.
  return decltype(std::get<I>(internal::declval<T>()))(1);
}

template <class T, class U, size_t... Is>
NDARRAY_HOST_DEVICE T convert_dims(const U& u, internal::index_sequence<Is...>) {
  return std::make_tuple(convert_dim<Is, T>(u)...);
}

constexpr index_t factorial(index_t x) { return x == 1 ? 1 : x * factorial(x - 1); }

// The errors that result from not satisfying this check are probably hell,
// but it would be pretty tricky to check that all of [0, Rank) is in `Is...`
template <size_t Rank, size_t... Is>
using enable_if_permutation = std::enable_if_t<sizeof...(Is) == Rank && all(Is < Rank...) &&
                                               product((Is + 2)...) == factorial(Rank + 1)>;

} // namespace internal

template <class... Dims>
class shape;

/** Helper function to make a tuple from a variadic list of `dims...`. */
template <class... Dims>
NDARRAY_HOST_DEVICE auto make_shape(Dims... dims) {
  return shape<Dims...>(dims...);
}

template <class... Dims>
NDARRAY_HOST_DEVICE shape<Dims...> make_shape_from_tuple(const std::tuple<Dims...>& dims) {
  return shape<Dims...>(dims);
}

/** Type of an index for an array of rank `Rank`. This will be
 * `std::tuple<...>` with `Rank` `index_t` values.
 *
 * For example, `index_of_rank<3>` is `std::tuple<index_t, index_t, index_t>`. */
template <size_t Rank>
using index_of_rank = internal::tuple_of_n<index_t, Rank>;

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

  // We use this a lot here. Make an alias for it.
  using dim_indices = decltype(internal::make_index_sequence<std::tuple_size<dims_type>::value>());

private:
  dims_type dims_;

  // TODO: This should use std::is_constructible<dims_type, std::tuple<OtherDims...>>
  // but it is broken on some compilers (https://github.com/dsharlet/array/issues/20).
  template <class... OtherDims>
  using enable_if_dims_compatible = std::enable_if_t<sizeof...(OtherDims) == rank()>;

  template <class... Args>
  using enable_if_same_rank = std::enable_if_t<(sizeof...(Args) == rank())>;

  template <class... Args>
  using enable_if_indices = std::enable_if_t<internal::all_of_type<index_t, Args...>::value>;

  template <class... Args>
  using enable_if_slices =
      std::enable_if_t<internal::all_of_any_type<std::tuple<interval<>, dim<>>, Args...>::value &&
                       !internal::all_of_type<index_t, Args...>::value>;

  template <size_t Dim>
  using enable_if_dim = std::enable_if_t<(Dim < rank())>;

public:
  NDARRAY_HOST_DEVICE shape() {}
  // TODO: This is a bit messy, but necessary to avoid ambiguous default
  // constructors when Dims is empty.
  template <size_t N = sizeof...(Dims), class = std::enable_if_t<(N > 0)>>
  NDARRAY_HOST_DEVICE shape(const Dims&... dims) : dims_(dims...) {}
  NDARRAY_HOST_DEVICE shape(const shape&) = default;
  NDARRAY_HOST_DEVICE shape(shape&&) = default;
  NDARRAY_HOST_DEVICE shape& operator=(const shape&) = default;
  NDARRAY_HOST_DEVICE shape& operator=(shape&&) = default;

  /** Construct or assign a shape from another set of dims of a possibly
   * different type. Each dim must be compatible with the corresponding
   * dim of this shape. */
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  NDARRAY_HOST_DEVICE shape(const std::tuple<OtherDims...>& other) : dims_(other) {}
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  NDARRAY_HOST_DEVICE shape(OtherDims... other_dims) : dims_(other_dims...) {}
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  NDARRAY_HOST_DEVICE shape(const shape<OtherDims...>& other) : dims_(other.dims()) {}
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  NDARRAY_HOST_DEVICE shape& operator=(const shape<OtherDims...>& other) {
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
  NDARRAY_HOST_DEVICE void resolve() { internal::resolve_unknown_strides(dims_, dim_indices()); }

  /** Check if all strides of the shape are known. */
  NDARRAY_HOST_DEVICE bool is_resolved() const {
    return internal::is_resolved(dims_, dim_indices());
  }

  /** Returns `true` if the indices or intervals `args` are in interval of this shape. */
  template <class... Args, class = enable_if_same_rank<Args...>>
  NDARRAY_HOST_DEVICE bool is_in_range(const std::tuple<Args...>& args) const {
    return internal::is_in_range(dims_, args, dim_indices());
  }
  template <class... Args, class = enable_if_same_rank<Args...>>
  NDARRAY_HOST_DEVICE bool is_in_range(Args... args) const {
    return internal::is_in_range(dims_, std::make_tuple(args...), dim_indices());
  }

  /** Compute the flat offset of the index `indices`. */
  NDARRAY_HOST_DEVICE index_t operator()(const index_type& indices) const {
    return internal::flat_offset(dims_, indices, dim_indices());
  }
  NDARRAY_HOST_DEVICE index_t operator[](const index_type& indices) const {
    return internal::flat_offset(dims_, indices, dim_indices());
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  NDARRAY_HOST_DEVICE index_t operator()(Args... indices) const {
    return internal::flat_offset(dims_, std::make_tuple(indices...), dim_indices());
  }

  /** Create a new shape from this shape using a indices or intervals `args`.
   * Dimensions corresponding to indices in `args` are sliced, i.e. the result
   * will not have this dimension. The rest of the dimensions are cropped. */
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  NDARRAY_HOST_DEVICE auto operator()(const std::tuple<Args...>& args) const {
    auto new_dims = internal::intervals_with_strides(args, dims_, dim_indices());
    auto new_dims_no_slices = internal::skip_slices(new_dims, args, dim_indices());
    return make_shape_from_tuple(new_dims_no_slices);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  NDARRAY_HOST_DEVICE auto operator[](const std::tuple<Args...>& args) const {
    return operator()(args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  NDARRAY_HOST_DEVICE auto operator()(Args... args) const {
    return operator()(std::make_tuple(args...));
  }

  /** Get a specific dim `D` of this shape. */
  template <size_t D, class = enable_if_dim<D>>
  NDARRAY_HOST_DEVICE auto& dim() {
    return std::get<D>(dims_);
  }
  template <size_t D, class = enable_if_dim<D>>
  NDARRAY_HOST_DEVICE const auto& dim() const {
    return std::get<D>(dims_);
  }

  /** Get a specific dim of this shape with a runtime dimension index `d`.
   * This will lose knowledge of any compile-time constant dimension
   * attributes. */
  NDARRAY_HOST_DEVICE nda::dim<> dim(size_t d) const {
    assert(d < rank());
    return internal::tuple_to_array<nda::dim<>>(dims_)[d];
  }

  /** Get a tuple of all of the dims of this shape. */
  NDARRAY_HOST_DEVICE dims_type& dims() { return dims_; }
  NDARRAY_HOST_DEVICE const dims_type& dims() const { return dims_; }

  NDARRAY_HOST_DEVICE index_type min() const { return internal::mins(dims(), dim_indices()); }
  NDARRAY_HOST_DEVICE index_type max() const { return internal::maxs(dims(), dim_indices()); }
  NDARRAY_HOST_DEVICE index_type extent() const { return internal::extents(dims(), dim_indices()); }
  NDARRAY_HOST_DEVICE index_type stride() const { return internal::strides(dims(), dim_indices()); }

  /** Compute the min, max, or extent of the flat offsets of this shape.
   * This is the extent of the valid interval of values returned by `operator()`
   * or `operator[]`. */
  NDARRAY_HOST_DEVICE index_t flat_min() const { return internal::flat_min(dims_, dim_indices()); }
  NDARRAY_HOST_DEVICE index_t flat_max() const { return internal::flat_max(dims_, dim_indices()); }
  NDARRAY_HOST_DEVICE size_type flat_extent() const {
    index_t e = flat_max() - flat_min() + 1;
    return e < 0 ? 0 : static_cast<size_type>(e);
  }

  /** Compute the total number of indices in this shape. */
  NDARRAY_HOST_DEVICE size_type size() const {
    index_t s = internal::product(extent(), dim_indices());
    return s < 0 ? 0 : static_cast<size_type>(s);
  }

  /** A shape is empty if its size is 0. */
  NDARRAY_HOST_DEVICE bool empty() const { return size() == 0; }

  /** Returns `true` if this shape is 'compact' in memory. A shape is compact
   * if there are no unaddressable flat indices between the first and last
   * addressable flat elements. */
  NDARRAY_HOST_DEVICE bool is_compact() const { return flat_extent() <= size(); }

  /** Returns `true` if this shape is an injective function mapping indices to
   * flat indices. If the dims overlap, or a dim has stride zero, multiple
   * indices will map to the same flat index; in this case, this function will
   * return `false`. */
  NDARRAY_HOST_DEVICE bool is_one_to_one() const {
    // TODO: https://github.com/dsharlet/array/issues/2
    return flat_extent() >= size();
  }

  /** Returns `true` if this shape projects to a set of flat indices that is a
   * subset of the other shape's projection to flat indices, with an offset
   * `offset`. */
  template <typename OtherShape>
  NDARRAY_HOST_DEVICE bool is_subset_of(const OtherShape& other, index_t offset) const {
    // TODO: https://github.com/dsharlet/array/issues/2
    return flat_min() >= other.flat_min() + offset && flat_max() <= other.flat_max() + offset;
  }

  /** Provide some aliases for common interpretations of dimensions
   * `i`, `j`, `k` as dimensions 0, 1, 2, respectively. */
  NDARRAY_HOST_DEVICE auto& i() { return dim<0>(); }
  NDARRAY_HOST_DEVICE const auto& i() const { return dim<0>(); }
  NDARRAY_HOST_DEVICE auto& j() { return dim<1>(); }
  NDARRAY_HOST_DEVICE const auto& j() const { return dim<1>(); }
  NDARRAY_HOST_DEVICE auto& k() { return dim<2>(); }
  NDARRAY_HOST_DEVICE const auto& k() const { return dim<2>(); }

  /** Provide some aliases for common interpretations of dimensions
   * `x`, `y`, `z` or `c`, `w` as dimensions 0, 1, 2, 3 respectively. */
  NDARRAY_HOST_DEVICE auto& x() { return dim<0>(); }
  NDARRAY_HOST_DEVICE const auto& x() const { return dim<0>(); }
  NDARRAY_HOST_DEVICE auto& y() { return dim<1>(); }
  NDARRAY_HOST_DEVICE const auto& y() const { return dim<1>(); }
  NDARRAY_HOST_DEVICE auto& z() { return dim<2>(); }
  NDARRAY_HOST_DEVICE const auto& z() const { return dim<2>(); }
  NDARRAY_HOST_DEVICE auto& c() { return dim<2>(); }
  NDARRAY_HOST_DEVICE const auto& c() const { return dim<2>(); }
  NDARRAY_HOST_DEVICE auto& w() { return dim<3>(); }
  NDARRAY_HOST_DEVICE const auto& w() const { return dim<3>(); }

  /** Assuming this array represents an image with dimensions {width,
   * height, channels}, get the extent of those dimensions. */
  NDARRAY_HOST_DEVICE index_t width() const { return x().extent(); }
  NDARRAY_HOST_DEVICE index_t height() const { return y().extent(); }
  NDARRAY_HOST_DEVICE index_t channels() const { return c().extent(); }

  /** Assuming this array represents a matrix with dimensions {rows,
   * cols}, get the extent of those dimensions. */
  NDARRAY_HOST_DEVICE index_t rows() const { return i().extent(); }
  NDARRAY_HOST_DEVICE index_t columns() const { return j().extent(); }

  /** A shape is equal to another shape if the dim objects of each
   * dimension from both shapes are equal. */
  template <class... OtherDims, class = enable_if_same_rank<OtherDims...>>
  NDARRAY_HOST_DEVICE bool operator==(const shape<OtherDims...>& other) const {
    return dims_ == other.dims();
  }
  template <class... OtherDims, class = enable_if_same_rank<OtherDims...>>
  NDARRAY_HOST_DEVICE bool operator!=(const shape<OtherDims...>& other) const {
    return dims_ != other.dims();
  }
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
NDARRAY_HOST_DEVICE auto transpose(const shape<Dims...>& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}
template <size_t... DimIndices, class... Dims>
NDARRAY_HOST_DEVICE auto reorder(const shape<Dims...>& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}

namespace internal {

template <class Fn, class Idx>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void for_each_index_in_order_impl(Fn&& fn, const Idx& idx) {
  fn(idx);
}

// These multiple dim at a time overloads help reduce pre-optimization
// code size.
template <class Fn, class OuterIdx, class Dim0>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_index_in_order_impl(
    Fn&& fn, const OuterIdx& idx, const Dim0& dim0) {
  for (index_t i : dim0) {
    fn(std::tuple_cat(std::tuple<index_t>(i), idx));
  }
}

template <class Fn, class OuterIdx, class Dim0, class Dim1>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_index_in_order_impl(
    Fn&& fn, const OuterIdx& idx, const Dim0& dim0, const Dim1& dim1) {
  for (index_t i : dim0) {
    for (index_t j : dim1) {
      fn(std::tuple_cat(std::tuple<index_t, index_t>(j, i), idx));
    }
  }
}

template <class Fn, class OuterIdx, class Dim0, class... Dims>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_index_in_order_impl(
    Fn&& fn, const OuterIdx& idx, const Dim0& dim0, const Dims&... dims) {
  for (index_t i : dim0) {
    for_each_index_in_order_impl(fn, std::tuple_cat(std::tuple<index_t>(i), idx), dims...);
  }
}

template <class Fn, class OuterIdx, class Dim0, class Dim1, class... Dims>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_index_in_order_impl(
    Fn&& fn, const OuterIdx& idx, const Dim0& dim0, const Dim1& dim1, const Dims&... dims) {
  for (index_t i : dim0) {
    for (index_t j : dim1) {
      for_each_index_in_order_impl(
          fn, std::tuple_cat(std::tuple<index_t, index_t>(j, i), idx), dims...);
    }
  }
}

template <class Dims, class Fn, size_t... Is>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void for_each_index_in_order(
    Fn&& fn, const Dims& dims, index_sequence<Is...>) {
  // We need to reverse the order of the dims so the last dim is
  // iterated innermost.
  for_each_index_in_order_impl(fn, std::tuple<>(), std::get<sizeof...(Is) - 1 - Is>(dims)...);
}

template <typename TSrc, typename TDst>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void move_assign(TSrc& src, TDst& dst) {
  dst = std::move(src);
}

template <typename TSrc, typename TDst>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void copy_assign(const TSrc& src, TDst& dst) {
  dst = src;
}

template <size_t D, class Ptr0>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void advance(Ptr0& ptr0) {
  std::get<0>(ptr0) += std::get<D>(std::get<1>(ptr0));
}
template <size_t D, class Ptr0, class Ptr1>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void advance(Ptr0& ptr0, Ptr1& ptr1) {
  std::get<0>(ptr0) += std::get<D>(std::get<1>(ptr0));
  std::get<0>(ptr1) += std::get<D>(std::get<1>(ptr1));
}
// If we ever need other than 1- or 2-way for_each_value, add a variadic
// version of advance.

template <class Fn, class... Ptrs>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_value_in_order_inner_dense(
    index_t extent, Fn&& fn, Ptrs NDARRAY_RESTRICT... ptrs) {
  for (index_t i = 0; i < extent; i++) {
    fn(*ptrs++...);
  }
}

template <size_t, class ExtentType, class Fn, class... Ptrs>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_value_in_order_impl(
    std::true_type, const ExtentType& extent, Fn&& fn, Ptrs... ptrs) {
  index_t extent_d = std::get<0>(extent);
  if (all(std::get<0>(std::get<1>(ptrs)) == 1 ...)) {
    for_each_value_in_order_inner_dense(extent_d, fn, std::get<0>(ptrs)...);
  } else {
    for (index_t i = 0; i < extent_d; i++) {
      fn(*std::get<0>(ptrs)...);
      advance<0>(ptrs...);
    }
  }
}

template <size_t D, class ExtentType, class Fn, class... Ptrs>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_value_in_order_impl(
    std::false_type, const ExtentType& extent, Fn&& fn, Ptrs... ptrs) {
  index_t extent_d = std::get<D>(extent);
  for (index_t i = 0; i < extent_d; i++) {
    using is_inner_loop = std::conditional_t<D == 1, std::true_type, std::false_type>;
    for_each_value_in_order_impl<D - 1>(is_inner_loop(), extent, fn, ptrs...);
    advance<D>(ptrs...);
  }
}

template <size_t D, class ExtentType, class Fn, class... Ptrs>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void for_each_value_in_order(
    const ExtentType& extent, Fn&& fn, Ptrs... ptrs) {
  using is_inner_loop = std::conditional_t<D == 0, std::true_type, std::false_type>;
  for_each_value_in_order_impl<D>(is_inner_loop(), extent, fn, ptrs...);
}

// Scalar buffers are a special case.
template <size_t D, class Fn, class... Ptrs>
NDARRAY_INLINE NDARRAY_HOST_DEVICE void for_each_value_in_order(
    const std::tuple<>& extent, Fn&& fn, Ptrs... ptrs) {
  fn(*std::get<0>(ptrs)...);
}

template <size_t Rank>
NDARRAY_HOST_DEVICE auto make_default_dense_shape() {
  // The inner dimension is a dense_dim, unless the shape is rank 0.
  using inner_dim = std::conditional_t<(Rank > 0), std::tuple<dense_dim<>>, std::tuple<>>;
  return make_shape_from_tuple(
      std::tuple_cat(inner_dim(), tuple_of_n<dim<>, std::max<size_t>(1, Rank) - 1>()));
}

template <index_t CurrentStride>
NDARRAY_HOST_DEVICE std::tuple<> make_compact_dims() {
  return std::tuple<>();
}

template <index_t CurrentStride, index_t Min, index_t Extent, index_t Stride, class... Dims>
NDARRAY_HOST_DEVICE auto make_compact_dims(
    const dim<Min, Extent, Stride>& dim0, const Dims&... dims) {
  // We already know the stride of this dimension.
  return std::tuple_cat(std::make_tuple(dim<Min, Extent, Stride>(dim0.min(), dim0.extent())),
      make_compact_dims<CurrentStride>(dims...));
}

template <index_t CurrentStride, index_t Min, index_t Extent, class... Dims>
NDARRAY_HOST_DEVICE auto make_compact_dims(const dim<Min, Extent>& dim0, const Dims&... dims) {
  // If we know the extent of this dimension, we can also provide
  // a constant stride for the next dimension.
  constexpr index_t NextStride = static_mul(CurrentStride, Extent);
  // Give this dimension the current stride, and don't give it a
  // runtime stride. If CurrentStride is static, that will be the
  // stride. If not, it will be dynamic, and resolved later.
  return std::tuple_cat(std::make_tuple(dim<Min, Extent, CurrentStride>(dim0.min(), dim0.extent())),
      make_compact_dims<NextStride>(dims...));
}

template <class Dims, size_t... Is>
NDARRAY_HOST_DEVICE auto make_compact_dims(const Dims& dims, index_sequence<Is...>) {
  // Currently, the algorithm is:
  // - If any dimension has a static stride greater than one, don't give any
  //   static strides.
  // - If all dimensions are dynamic, use 1 as the next stride.
  // - If a dimension has static stride 1, use its Extent as the next stride.
  // This is very conservative, but the only thing I can figure out how to
  // express as constexpr with C++14 that doesn't risk doing dumb things.
  constexpr index_t MinStride =
      variadic_max(std::tuple_element<Is, Dims>::type::Stride == 1
                       ? static_abs(std::tuple_element<Is, Dims>::type::Extent)
                       : dynamic...);
  constexpr bool AnyStrideGreaterThanOne = any((std::tuple_element<Is, Dims>::type::Stride > 1)...);
  constexpr bool AllDynamic = all(is_dynamic(std::tuple_element<Is, Dims>::type::Stride)...);
  constexpr index_t NextStride = AnyStrideGreaterThanOne ? dynamic : (AllDynamic ? 1 : MinStride);
  return make_compact_dims<NextStride>(std::get<Is>(dims)...);
}

template <index_t Min, index_t Extent, index_t Stride, class DimSrc>
NDARRAY_HOST_DEVICE bool is_dim_compatible(const dim<Min, Extent, Stride>&, const DimSrc& src) {
  return (is_dynamic(Min) || src.min() == Min) && (is_dynamic(Extent) || src.extent() == Extent) &&
         (is_dynamic(Stride) || src.stride() == Stride);
}

template <class... DimsDst, class ShapeSrc, size_t... Is>
NDARRAY_HOST_DEVICE bool is_shape_compatible(
    const shape<DimsDst...>&, const ShapeSrc& src, index_sequence<Is...>) {
  return all(is_dim_compatible(DimsDst(), src.template dim<Is>())...);
}

template <class DimA, class DimB>
NDARRAY_HOST_DEVICE auto clamp_dims(const DimA& a, const DimB& b) {
  constexpr index_t Min = static_max(DimA::Min, DimB::Min);
  constexpr index_t Max = static_min(DimA::Max, DimB::Max);
  constexpr index_t Extent = static_add(static_sub(Max, Min), 1);
  index_t min = std::max(a.min(), b.min());
  index_t max = std::min(a.max(), b.max());
  index_t extent = max - min + 1;
  return dim<Min, Extent, DimA::Stride>(min, extent, a.stride());
}

template <class DimsA, class DimsB, size_t... Is>
NDARRAY_HOST_DEVICE auto clamp(const DimsA& a, const DimsB& b, index_sequence<Is...>) {
  return make_shape(clamp_dims(std::get<Is>(a), std::get<Is>(b))...);
}

// Return where the index I appears in Is...
template <size_t I>
constexpr size_t index_of() {
  // This constant is just something that would be absurd to use as a tuple
  // index, but has headroom so we can add to it without overflow.
  return 10000;
}
template <size_t I, size_t I0, size_t... Is>
constexpr size_t index_of() {
  return I == I0 ? 0 : 1 + index_of<I, Is...>();
}

// Similar to std::get, but returns a one-element tuple if I is
// in bounds, or an empty tuple if not.
template <size_t I, class T, std::enable_if_t<(I < std::tuple_size<T>::value), int> = 0>
NDARRAY_INLINE NDARRAY_HOST_DEVICE auto get_or_empty(const T& t) {
  return std::make_tuple(std::get<I>(t));
}
template <size_t I, class T, std::enable_if_t<(I >= std::tuple_size<T>::value), int> = 0>
NDARRAY_INLINE NDARRAY_HOST_DEVICE std::tuple<> get_or_empty(const T& t) {
  return std::tuple<>();
}

// Perform the inverse of a shuffle with indices Is...
template <size_t... Is, class T, size_t... Js>
NDARRAY_HOST_DEVICE auto unshuffle(const T& t, index_sequence<Js...>) {
  return std::tuple_cat(get_or_empty<index_of<Js, Is...>()>(t)...);
}
template <size_t... Is, class... Ts>
NDARRAY_HOST_DEVICE auto unshuffle(const std::tuple<Ts...>& t) {
  return unshuffle<Is...>(t, make_index_sequence<variadic_max(Is...) + 1>());
}

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_compatible =
    std::enable_if_t<std::is_constructible<ShapeDst, ShapeSrc>::value>;

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_explicitly_compatible =
    std::enable_if_t<(ShapeSrc::rank() <= ShapeSrc::rank())>;

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_copy_compatible = std::enable_if_t<(ShapeDst::rank() == ShapeSrc::rank())>;

template <class Alloc>
using enable_if_allocator = decltype(internal::declval<Alloc>().allocate(0));

} // namespace internal

/** An arbitrary `shape` with the specified rank `Rank`. This shape is
 * compatible with any other shape of the same rank. */
template <size_t Rank>
using shape_of_rank = decltype(make_shape_from_tuple(internal::tuple_of_n<dim<>, Rank>()));

/** A `shape` where the innermost dimension is a `dense_dim`, and all other
 * dimensions are arbitrary. */
template <size_t Rank>
using dense_shape = decltype(internal::make_default_dense_shape<Rank>());

/** Attempt to make both the compile-time and run-time strides of `s`
 * compact such that there is no padding between dimensions. Only dynamic
 * strides are potentially replaced with static strides, existing
 * compile-time strides are not modified. Run-time strides are then
 * populated using the `shape::resolve` algorithm.
 *
 * For a shape without any existing constant strides, this will return
 * an instance of `dense_shape<Shape::rank()>`.
 *
 * The resulting shape may not have `Shape::is_compact` return `true`
 * if the shape has existing non-compact compile-time constant strides. */
template <class Shape>
NDARRAY_HOST_DEVICE auto make_compact(const Shape& s) {
  auto static_compact =
      make_shape_from_tuple(internal::make_compact_dims(s.dims(), typename Shape::dim_indices()));
  static_compact.resolve();
  return static_compact;
}

/** Returns `true` if a shape `src` can be assigned to a shape of type
 * `ShapeDst` without error. */
template <class ShapeDst, class ShapeSrc,
    class = internal::enable_if_shapes_compatible<ShapeSrc, ShapeDst>>
NDARRAY_HOST_DEVICE bool is_compatible(const ShapeSrc& src) {
  return internal::is_shape_compatible(ShapeDst(), src, typename ShapeSrc::dim_indices());
}

/** Convert a shape `src` to shape type `ShapeDst`. This explicit conversion
 * allows converting a low rank shape to a higher ranked shape, where new
 * dimensions have min 0 and extent 1. */
// TODO: Consider enabling this kind of conversion implicitly. It is hard to
// do without constructor overload ambiguity problems, and I'm also not sure
// it's a good idea.
template <class ShapeDst, class ShapeSrc,
    class = internal::enable_if_shapes_explicitly_compatible<ShapeDst, ShapeSrc>>
NDARRAY_HOST_DEVICE ShapeDst convert_shape(const ShapeSrc& src) {
  return internal::convert_dims<typename ShapeDst::dims_type>(
      src.dims(), typename ShapeDst::dim_indices());
}

/** Test if a shape `src` can be explicitly converted to a shape of type
 * `ShapeDst` using `convert_shape` without error. */
template <class ShapeDst, class ShapeSrc,
    class = internal::enable_if_shapes_explicitly_compatible<ShapeSrc, ShapeDst>>
NDARRAY_HOST_DEVICE bool is_explicitly_compatible(const ShapeSrc& src) {
  return internal::is_shape_compatible(ShapeDst(), src, typename ShapeSrc::dim_indices());
}

/** Iterate over all indices in the shape, calling a function `fn` for each set
 * of indices. The indices are in the same order as the dims in the shape. The
 * first dim is the 'inner' loop of the iteration, and the last dim is the
 * 'outer' loop.
 *
 * These functions are typically used to implement `shape_traits<>` and
 * `copy_shape_traits<>` objects. Use `for_each_index`,
 * `array_ref<>::for_each_value`, or `array<>::for_each_value` instead. */
template <class Shape, class Fn,
    class = internal::enable_if_callable<Fn, typename Shape::index_type>>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_index_in_order(const Shape& shape, Fn&& fn) {
  internal::for_each_index_in_order(fn, shape.dims(), typename Shape::dim_indices());
}
template <class Shape, class Ptr, class Fn,
    class = internal::enable_if_callable<Fn, typename std::remove_pointer<Ptr>::type&>>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_value_in_order(
    const Shape& shape, Ptr base, Fn&& fn) {
  // TODO: This is losing compile-time constant extents and strides info
  // (https://github.com/dsharlet/array/issues/1).
  auto base_and_stride = std::make_pair(base, shape.stride());
  internal::for_each_value_in_order<Shape::rank() - 1>(shape.extent(), fn, base_and_stride);
}

/** Similar to `for_each_value_in_order`, but iterates over two arrays
 * simultaneously. `shape` defines the loop nest, while `shape_a` and `shape_b`
 * define the memory layout of `base_a` and `base_b`. */
template <class Shape, class ShapeA, class PtrA, class ShapeB, class PtrB, class Fn,
    class = internal::enable_if_callable<Fn, typename std::remove_pointer<PtrA>::type&,
        typename std::remove_pointer<PtrB>::type&>>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_value_in_order(const Shape& shape,
    const ShapeA& shape_a, PtrA base_a, const ShapeB& shape_b, PtrB base_b, Fn&& fn) {
  base_a += shape_a(shape.min());
  base_b += shape_b(shape.min());
  // TODO: This is losing compile-time constant extents and strides info
  // (https://github.com/dsharlet/array/issues/1).
  auto a = std::make_pair(base_a, shape_a.stride());
  auto b = std::make_pair(base_b, shape_b.stride());
  internal::for_each_value_in_order<Shape::rank() - 1>(shape.extent(), fn, a, b);
}

namespace internal {

NDARRAY_INLINE NDARRAY_HOST_DEVICE bool can_fuse(const dim<>& inner, const dim<>& outer) {
  return inner.stride() * inner.extent() == outer.stride();
}

NDARRAY_INLINE NDARRAY_HOST_DEVICE dim<> fuse(const dim<>& inner, const dim<>& outer) {
  assert(can_fuse(inner, outer));
  return dim<>(
      inner.min() + outer.min() * inner.extent(), inner.extent() * outer.extent(), inner.stride());
}

struct copy_dims {
  dim<> src;
  dim<> dst;
};

inline bool operator<(const dim<>& l, const dim<>& r) { return l.stride() < r.stride(); }

inline bool operator<(const copy_dims& l, const copy_dims& r) {
  return l.dst.stride() < r.dst.stride();
}

// We need a sort that only needs to deal with very small lists,
// and extra complexity here is costly in code size/compile time.
// This is a rare job for bubble sort!
template <class Iterator>
NDARRAY_HOST_DEVICE void bubble_sort(Iterator begin, Iterator end) {
  for (Iterator i = begin; i != end; ++i) {
    for (Iterator j = i; j != end; ++j) {
      if (*j < *i) { std::swap(*i, *j); }
    }
  }
}

// Sort the dims such that strides are increasing from dim 0, and contiguous
// dimensions are fused.
template <class Shape>
NDARRAY_HOST_DEVICE shape_of_rank<Shape::rank()> dynamic_optimize_shape(const Shape& shape) {
  auto dims = internal::tuple_to_array<dim<>>(shape.dims());

  // Sort the dims by stride.
  bubble_sort(dims.begin(), dims.end());

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
    dims[i] = dim<>(0, 1, 0);
  }

  return shape_of_rank<Shape::rank()>(array_to_tuple(dims));
}

// Optimize a src and dst shape. The dst shape is made dense, and contiguous
// dimensions are fused.
template <class ShapeSrc, class ShapeDst,
    class = enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
NDARRAY_HOST_DEVICE auto dynamic_optimize_copy_shapes(const ShapeSrc& src, const ShapeDst& dst) {
  constexpr size_t rank = ShapeSrc::rank();
  static_assert(rank == ShapeDst::rank(), "copy shapes must have same rank.");
  auto src_dims = internal::tuple_to_array<dim<>>(src.dims());
  auto dst_dims = internal::tuple_to_array<dim<>>(dst.dims());

  std::array<copy_dims, rank> dims;
  for (size_t i = 0; i < rank; i++) {
    dims[i] = {src_dims[i], dst_dims[i]};
  }

  // Sort the dims by the dst stride.
  bubble_sort(dims.begin(), dims.end());

  // Find dimensions that are contiguous and fuse them.
  size_t new_rank = dims.size();
  for (size_t i = 0; i + 1 < new_rank;) {
    if (dims[i].src.extent() == dims[i].dst.extent() && can_fuse(dims[i].src, dims[i + 1].src) &&
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
    dims[i] = {dim<>(0, 1, 0), dim<>(0, 1, 0)};
  }

  for (size_t i = 0; i < dims.size(); i++) {
    src_dims[i] = dims[i].src;
    dst_dims[i] = dims[i].dst;
  }

  return std::make_pair(
      shape_of_rank<rank>(array_to_tuple(src_dims)), shape_of_rank<rank>(array_to_tuple(dst_dims)));
}

template <class Shape>
NDARRAY_HOST_DEVICE auto optimize_shape(const Shape& shape) {
  // In the general case, dynamically optimize the shape.
  return dynamic_optimize_shape(shape);
}

template <class Dim0>
NDARRAY_HOST_DEVICE auto optimize_shape(const shape<Dim0>& shape) {
  // Nothing to do for rank 1 shapes.
  return shape;
}

template <class ShapeSrc, class ShapeDst>
NDARRAY_HOST_DEVICE auto optimize_copy_shapes(const ShapeSrc& src, const ShapeDst& dst) {
  return dynamic_optimize_copy_shapes(src, dst);
}

template <class Dim0Src, class Dim0Dst>
NDARRAY_HOST_DEVICE auto optimize_copy_shapes(
    const shape<Dim0Src>& src, const shape<Dim0Dst>& dst) {
  // Nothing to do for rank 1 shapes.
  return std::make_pair(src, dst);
}

template <class T>
NDARRAY_HOST_DEVICE T* pointer_add(T* x, index_t offset) {
  return x != nullptr ? x + offset : x;
}

} // namespace internal

/** Shape traits enable some behaviors to be customized per shape type. */
template <class Shape>
class shape_traits {
public:
  using shape_type = Shape;

  /** The `for_each_index` implementation for the shape may choose to iterate in a
   * different order than the default (in-order). */
  template <class Fn>
  NDARRAY_HOST_DEVICE static void for_each_index(const Shape& shape, Fn&& fn) {
    for_each_index_in_order(shape, fn);
  }

  /** The `for_each_value` implementation for the shape may be able to statically
   * optimize shape. The default implementation optimizes the shape at runtime,
   * and only attempts to convert the shape to a `dense_shape`. */
  template <class Ptr, class Fn>
  NDARRAY_HOST_DEVICE static void for_each_value(const Shape& shape, Ptr base, Fn&& fn) {
    auto opt_shape = internal::optimize_shape(shape);
    for_each_value_in_order(opt_shape, base, fn);
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
  NDARRAY_HOST_DEVICE static void for_each_value(
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
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_index(const Shape& s, Fn&& fn) {
  shape_traits<Shape>::for_each_index(s, fn);
}
template <size_t... LoopOrder, class Shape, class Fn,
    class = internal::enable_if_applicable<Fn, typename Shape::index_type>,
    std::enable_if_t<(sizeof...(LoopOrder) == 0), int> = 0>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_all_indices(const Shape& s, Fn&& fn) {
  using index_type = typename Shape::index_type;
  for_each_index(s, [fn = std::move(fn)](const index_type& i) { internal::apply(fn, i); });
}
template <size_t... LoopOrder, class Shape, class Fn,
    class = internal::enable_if_callable<Fn, index_of_rank<sizeof...(LoopOrder)>>,
    std::enable_if_t<(sizeof...(LoopOrder) != 0), int> = 0>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_each_index(const Shape& s, Fn&& fn) {
  using index_type = index_of_rank<sizeof...(LoopOrder)>;
  for_each_index_in_order(reorder<LoopOrder...>(s),
      [fn = std::move(fn)](const index_type& i) { fn(internal::unshuffle<LoopOrder...>(i)); });
}
template <size_t... LoopOrder, class Shape, class Fn,
    class = internal::enable_if_callable<Fn, decltype(LoopOrder)...>,
    std::enable_if_t<(sizeof...(LoopOrder) != 0), int> = 0>
NDARRAY_UNIQUE NDARRAY_HOST_DEVICE void for_all_indices(const Shape& s, Fn&& fn) {
  using index_type = index_of_rank<sizeof...(LoopOrder)>;
  for_each_index_in_order(reorder<LoopOrder...>(s), [fn = std::move(fn)](const index_type& i) {
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
NDARRAY_HOST_DEVICE array_ref<T, Shape> make_array_ref(T* base, const Shape& shape) {
  return {base, shape};
}

namespace internal {

template <class T, class Shape>
NDARRAY_HOST_DEVICE array_ref<T, Shape> make_array_ref_no_resolve(T* base, const Shape& shape) {
  return {base, shape, std::false_type()};
}

template <class T, class Shape, class... Args>
NDARRAY_HOST_DEVICE auto make_array_ref_at(
    T base, const Shape& shape, const std::tuple<Args...>& args) {
  auto new_shape = shape(args);
  auto new_mins = mins_of_intervals(args, shape.dims(), make_index_sequence<sizeof...(Args)>());
  auto old_min_offset = shape(new_mins);
  return make_array_ref_no_resolve(internal::pointer_add(base, old_min_offset), new_shape);
}

} // namespace internal

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
  using enable_if_same_rank = std::enable_if_t<sizeof...(Args) == rank()>;

  template <class... Args>
  using enable_if_indices = std::enable_if_t<internal::all_of_type<index_t, Args...>::value>;

  template <class... Args>
  using enable_if_slices =
      std::enable_if_t<internal::all_of_any_type<std::tuple<interval<>, dim<>>, Args...>::value &&
                       !internal::all_of_type<index_t, Args...>::value>;

  template <size_t Dim>
  using enable_if_dim = std::enable_if_t < Dim<rank()>;

  pointer base_;
  Shape shape_;

public:
  /** Make an array_ref to the given `base` pointer, interpreting it as having
   * the shape `shape`. */
  NDARRAY_HOST_DEVICE array_ref(pointer base = nullptr, const Shape& shape = Shape())
      : base_(base), shape_(shape) {
    shape_.resolve();
  }

  NDARRAY_HOST_DEVICE array_ref(pointer base, const Shape& shape, std::false_type /*resolve*/)
      : base_(base), shape_(shape) {}

  /** Shallow copy or assign an array_ref. */
  NDARRAY_HOST_DEVICE array_ref(const array_ref& other) = default;
  NDARRAY_HOST_DEVICE array_ref(array_ref&& other) = default;
  NDARRAY_HOST_DEVICE array_ref& operator=(const array_ref& other) = default;
  NDARRAY_HOST_DEVICE array_ref& operator=(array_ref&& other) = default;

  /** Shallow copy or assign an array_ref with a different shape type. */
  template <class OtherShape, class = enable_if_shape_compatible<OtherShape>>
  NDARRAY_HOST_DEVICE array_ref(const array_ref<T, OtherShape>& other)
      : array_ref(other.base(), other.shape(), std::false_type()) {}
  template <class OtherShape, class = enable_if_shape_compatible<OtherShape>>
  NDARRAY_HOST_DEVICE array_ref& operator=(const array_ref<T, OtherShape>& other) {
    base_ = other.base();
    shape_ = other.shape();
    return *this;
  }

  /** Get a reference to the element at `indices`. */
  NDARRAY_HOST_DEVICE reference operator()(const index_type& indices) const {
    return base_[shape_(indices)];
  }
  NDARRAY_HOST_DEVICE reference operator[](const index_type& indices) const {
    return base_[shape_(indices)];
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  NDARRAY_HOST_DEVICE reference operator()(Args... indices) const {
    return base_[shape_(indices...)];
  }

  /** Create an `array_ref` from this array_ref using a indices or intervals
   * `args`. Dimensions corresponding to indices in `args` are sliced, i.e.
   * the result will not have this dimension. The rest of the dimensions are
   * cropped. */
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  NDARRAY_HOST_DEVICE auto operator()(const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  NDARRAY_HOST_DEVICE auto operator[](const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  NDARRAY_HOST_DEVICE auto operator()(Args... args) const {
    return internal::make_array_ref_at(base_, shape_, std::make_tuple(args...));
  }

  /** Call a function with a reference to each value in this array_ref. The
   * order in which `fn` is called is undefined, to enable optimized memory
   * access patterns. */
  template <class Fn, class = internal::enable_if_callable<Fn, reference>>
  NDARRAY_HOST_DEVICE void for_each_value(Fn&& fn) const {
    shape_traits_type::for_each_value(shape_, base_, fn);
  }

  /** Pointer to the element at the min index of the shape. */
  NDARRAY_HOST_DEVICE pointer base() const { return base_; }

  /** Pointer to the element at the beginning of the flat array. This is
   * equivalent to `base()` if all of the strides of the shape are positive. */
  NDARRAY_HOST_DEVICE pointer data() const {
    return internal::pointer_add(base_, shape_.flat_min());
  }

  /** Shape of this array_ref. */
  NDARRAY_HOST_DEVICE Shape& shape() { return shape_; }
  NDARRAY_HOST_DEVICE const Shape& shape() const { return shape_; }

  template <size_t D, class = enable_if_dim<D>>
  NDARRAY_HOST_DEVICE auto& dim() {
    return shape_.template dim<D>();
  }
  template <size_t D, class = enable_if_dim<D>>
  NDARRAY_HOST_DEVICE const auto& dim() const {
    return shape_.template dim<D>();
  }
  NDARRAY_HOST_DEVICE size_type size() const { return shape_.size(); }
  NDARRAY_HOST_DEVICE bool empty() const { return base() != nullptr ? shape_.empty() : true; }
  NDARRAY_HOST_DEVICE bool is_compact() const { return shape_.is_compact(); }

  /** Provide some aliases for common interpretations of dimensions
   * `i`, `j`, `k` as dimensions 0, 1, 2, respectively. */
  NDARRAY_HOST_DEVICE auto& i() { return shape_.i(); }
  NDARRAY_HOST_DEVICE const auto& i() const { return shape_.i(); }
  NDARRAY_HOST_DEVICE auto& j() { return shape_.j(); }
  NDARRAY_HOST_DEVICE const auto& j() const { return shape_.j(); }
  NDARRAY_HOST_DEVICE auto& k() { return shape_.k(); }
  NDARRAY_HOST_DEVICE const auto& k() const { return shape_.k(); }

  /** Provide some aliases for common interpretations of dimensions
   * `x`, `y`, `z` or `c`, `w` as dimensions 0, 1, 2, 3 respectively. */
  NDARRAY_HOST_DEVICE auto& x() { return shape_.x(); }
  NDARRAY_HOST_DEVICE const auto& x() const { return shape_.x(); }
  NDARRAY_HOST_DEVICE auto& y() { return shape_.y(); }
  NDARRAY_HOST_DEVICE const auto& y() const { return shape_.y(); }
  NDARRAY_HOST_DEVICE auto& z() { return shape_.z(); }
  NDARRAY_HOST_DEVICE const auto& z() const { return shape_.z(); }
  NDARRAY_HOST_DEVICE auto& c() { return shape_.c(); }
  NDARRAY_HOST_DEVICE const auto& c() const { return shape_.c(); }
  NDARRAY_HOST_DEVICE auto& w() { return shape_.w(); }
  NDARRAY_HOST_DEVICE const auto& w() const { return shape_.w(); }

  /** Assuming this array represents an image with dimensions width, height,
   * channels, get the extent of those dimensions. */
  NDARRAY_HOST_DEVICE index_t width() const { return shape_.width(); }
  NDARRAY_HOST_DEVICE index_t height() const { return shape_.height(); }
  NDARRAY_HOST_DEVICE index_t channels() const { return shape_.channels(); }

  /** Assuming this array represents a matrix with dimensions {rows, cols}, get
   * the extent of those dimensions. */
  NDARRAY_HOST_DEVICE index_t rows() const { return shape_.rows(); }
  NDARRAY_HOST_DEVICE index_t columns() const { return shape_.columns(); }

  /** Compare the contents of this array_ref to `other`. For two array_refs to
   * be considered equal, they must have the same shape, and all elements
   * addressable by the shape must also be equal. */
  // TODO: Maybe this should just check for equality of the shape and pointer,
  // and let the free function equal serve this purpose.
  NDARRAY_HOST_DEVICE bool operator!=(const array_ref& other) const {
    if (shape_ != other.shape_) { return true; }

    // TODO: This currently calls operator!= on all elements of the array_ref,
    // even after we find a non-equal element
    // (https://github.com/dsharlet/array/issues/4).
    bool result = false;
    copy_shape_traits<Shape, Shape>::for_each_value(
        shape_, base_, other.shape_, other.base_, [&](const_reference a, const_reference b) {
          if (a != b) { result = true; }
        });
    return result;
  }
  NDARRAY_HOST_DEVICE bool operator==(const array_ref& other) const { return !operator!=(other); }

  NDARRAY_HOST_DEVICE const array_ref<T, Shape>& ref() const { return *this; }

  /** Allow conversion from array_ref<T> to const_array_ref<T>. */
  NDARRAY_HOST_DEVICE const const_array_ref<T, Shape> cref() const {
    return const_array_ref<T, Shape>(base_, shape_);
  }
  NDARRAY_HOST_DEVICE operator const_array_ref<T, Shape>() const { return cref(); }

  /** Implicit conversion to `T` for scalar shaped array_refs. */
  NDARRAY_HOST_DEVICE operator reference() const {
    static_assert(rank() == 0, "Cannot convert non-scalar array to scalar.");
    return *base_;
  }

  /** Change the shape of the array to `new_shape`, and move the base pointer
   * by `offset`. The new shape must be a subset of the old shape. */
  NDARRAY_HOST_DEVICE void set_shape(const Shape& new_shape, index_t offset = 0) {
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
  using enable_if_same_rank = std::enable_if_t<sizeof...(Args) == rank()>;

  template <class... Args>
  using enable_if_indices = std::enable_if_t<internal::all_of_type<index_t, Args...>::value>;

  template <class... Args>
  using enable_if_slices =
      std::enable_if_t<internal::all_of_any_type<std::tuple<interval<>, dim<>>, Args...>::value &&
                       !internal::all_of_type<index_t, Args...>::value>;

  template <size_t Dim>
  using enable_if_dim = std::enable_if_t < Dim<rank()>;

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
    for_each_value([&](T& x) { alloc_traits::construct(alloc_, &x); });
  }
  void construct(const T& init) {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) { alloc_traits::construct(alloc_, &x, init); });
  }
  void copy_construct(const array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape_);
    copy_shape_traits_type::for_each_value(other.shape_, other.base_, shape_, base_,
        [&](const_reference src, reference dst) { alloc_traits::construct(alloc_, &dst, src); });
  }
  void move_construct(array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape_);
    copy_shape_traits_type::for_each_value(
        other.shape_, other.base_, shape_, base_, [&](reference src, reference dst) {
          alloc_traits::construct(alloc_, &dst, std::move(src));
        });
  }

  // Call the dstructor on every element.
  void destroy() {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) { alloc_traits::destroy(alloc_, &x); });
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
  array(const array& other, const Alloc& alloc) : array(alloc) { assign(other); }

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
  ~array() { deallocate(); }

  // Let's choose not to provide array(const array_ref&) constructors. This is
  // a deep copy that may be unintentional, perhaps it is better to require
  // being explicit via `make_copy`.

  /** Assign the contents of the array as a copy of `other`. The array is
   * deallocated if the allocator cannot be propagated on assignment. The array
   * is then reallocated if necessary, and each element in the array is copy
   * constructed from other. */
  array& operator=(const array& other) {
    if (base_ == other.base_) {
      if (base_) {
        assert(shape_ == other.shape_);
      } else {
        shape_ = other.shape_;
        assert(shape_.empty());
      }
      return *this;
    }

    if (alloc_traits::propagate_on_container_copy_assignment::value) {
      if (alloc_ != other.get_allocator()) { deallocate(); }
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
    if (base_ == other.base_) {
      if (base_) {
        assert(shape_ == other.shape_);
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
    if (base_ == other.base_) {
      if (base_) {
        assert(shape_ == other.shape_);
      } else {
        shape_ = other.shape_;
        assert(shape_.empty());
      }
      return;
    }
    if (shape_ == other.shape_) {
      destroy();
    } else {
      deallocate();
      shape_ = other.shape_;
      allocate();
    }
    copy_construct(other);
  }
  void assign(array&& other) {
    if (base_ == other.base_) {
      if (base_) {
        assert(shape_ == other.shape_);
      } else {
        shape_ = other.shape_;
        assert(shape_.empty());
      }
      return;
    }
    if (shape_ == other.shape_) {
      destroy();
    } else {
      deallocate();
      shape_ = other.shape_;
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
  reference operator()(const index_type& indices) { return base_[shape_(indices)]; }
  reference operator[](const index_type& indices) { return base_[shape_(indices)]; }
  const_reference operator()(const index_type& indices) const { return base_[shape_(indices)]; }
  const_reference operator[](const index_type& indices) const { return base_[shape_(indices)]; }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  reference operator()(Args... indices) {
    return base_[shape_(indices...)];
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_indices<Args...>>
  const_reference operator()(Args... indices) const {
    return base_[shape_(indices...)];
  }

  /** Create an `array_ref` from this array using a indices or intervals `args`.
   * Dimensions corresponding to indices in `args` are sliced, i.e. the result
   * will not have this dimension. The rest of the dimensions are cropped. */
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator()(const std::tuple<Args...>& args) {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator[](const std::tuple<Args...>& args) {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator()(Args... args) {
    return internal::make_array_ref_at(base_, shape_, std::make_tuple(args...));
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator()(const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator[](const std::tuple<Args...>& args) const {
    return internal::make_array_ref_at(base_, shape_, args);
  }
  template <class... Args, class = enable_if_same_rank<Args...>, class = enable_if_slices<Args...>>
  auto operator()(Args... args) const {
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

  /** Pointer to the element at the beginning of the flat array. This is
   * equivalent to `base()` if all of the strides of the shape are positive. */
  pointer data() { return internal::pointer_add(base_, shape_.flat_min()); }
  const_pointer data() const { return internal::pointer_add(base_, shape_.flat_min()); }

  /** Shape of this array. */
  const Shape& shape() const { return shape_; }

  template <size_t D, class = enable_if_dim<D>>
  const auto& dim() const {
    return shape_.template dim<D>();
  }
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
    if (shape_ == new_shape) { return; }

    // Allocate an array with the new shape.
    array new_array(new_shape);

    // Move the common elements to the new array.
    Shape intersection =
        internal::clamp(new_shape.dims(), shape_.dims(), typename Shape::dim_indices());
    pointer intersection_base =
        internal::pointer_add(new_array.base_, new_shape(intersection.min()));
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

  /** Implicit conversion to `T` for scalar shaped arrays. */
  operator reference() {
    static_assert(rank() == 0, "Cannot convert non-scalar array to scalar.");
    return *base_;
  }
  operator const_reference() const {
    static_assert(rank() == 0, "Cannot convert non-scalar array to scalar.");
    return *base_;
  }
};

/** An array type with an arbitrary shape of rank `Rank`. */
template <class T, size_t Rank, class Alloc = std::allocator<T>>
using array_of_rank = array<T, shape_of_rank<Rank>, Alloc>;

/** An array type with a shape `dense_shape<Rank>`. */
template <class T, size_t Rank, class Alloc = std::allocator<T>>
using dense_array = array<T, dense_shape<Rank>, Alloc>;

/** Make a new array with shape `shape`, allocated using `alloc`. */
template <class T, class Shape, class Alloc = std::allocator<T>,
    class = internal::enable_if_allocator<Alloc>>
auto make_array(const Shape& shape, const Alloc& alloc = Alloc()) {
  return array<T, Shape, Alloc>(shape, alloc);
}
template <class T, class Shape, class Alloc = std::allocator<T>,
    class = internal::enable_if_allocator<Alloc>>
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
  if (dst.shape().empty()) { return; }

  assert(src.shape().is_in_range(dst.shape().min()) && src.shape().is_in_range(dst.shape().max()));

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
template <class T, class ShapeSrc, class ShapeDst,
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
auto make_copy(const array<T, ShapeSrc, AllocSrc>& src, const ShapeDst& shape,
    const AllocDst& alloc = AllocDst()) {
  return make_copy(src.cref(), shape, alloc);
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
  if (dst.shape().empty()) { return; }

  assert(src.shape().is_in_range(dst.shape().min()) && src.shape().is_in_range(dst.shape().max()));

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
void move(array<T, Shape, Alloc>&& src, array<T, Shape, Alloc>& dst) {
  dst = std::move(src);
}

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
NDARRAY_HOST_DEVICE void fill(const array_ref<T, Shape>& dst, const T& value) {
  dst.for_each_value([value](T& i) { i = value; });
}
template <class T, class Shape, class Alloc>
void fill(array<T, Shape, Alloc>& dst, const T& value) {
  fill(dst.ref(), value);
}

/** Fill `dst` array or array_ref with the result of calling a generator
 * `g`. The order in which `g` is called is the same as
 * `shape_traits<Shape>::for_each_value`. */
template <class T, class Shape, class Generator, class = internal::enable_if_callable<Generator>>
NDARRAY_HOST_DEVICE void generate(const array_ref<T, Shape>& dst, Generator&& g) {
  dst.for_each_value([g = std::move(g)](T& i) { i = g(); });
}
template <class T, class Shape, class Alloc, class Generator,
    class = internal::enable_if_callable<Generator>>
void generate(array<T, Shape, Alloc>& dst, Generator&& g) {
  generate(dst.ref(), g);
}

/** Check if two array or array_refs have equal contents. */
template <class TA, class ShapeA, class TB, class ShapeB>
NDARRAY_HOST_DEVICE bool equal(const array_ref<TA, ShapeA>& a, const array_ref<TB, ShapeB>& b) {
  if (a.shape().min() != b.shape().min() || a.shape().extent() != b.shape().extent()) {
    return false;
  }

  bool result = true;
  copy_shape_traits<ShapeA, ShapeB>::for_each_value(
      a.shape(), a.base(), b.shape(), b.base(), [&](const TA& a, const TB& b) {
        if (a != b) { result = false; }
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
NDARRAY_HOST_DEVICE array_ref<T, NewShape> convert_shape(const array_ref<T, OldShape>& a) {
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
template <class U, class T, class Shape, class = std::enable_if_t<sizeof(T) == sizeof(U)>>
NDARRAY_HOST_DEVICE array_ref<U, Shape> reinterpret(const array_ref<T, Shape>& a) {
  return array_ref<U, Shape>(reinterpret_cast<U*>(a.base()), a.shape());
}
template <class U, class T, class Shape, class Alloc,
    class = std::enable_if_t<sizeof(T) == sizeof(U)>>
array_ref<U, Shape> reinterpret(array<T, Shape, Alloc>& a) {
  return reinterpret<U>(a.ref());
}
template <class U, class T, class Shape, class Alloc,
    class = std::enable_if_t<sizeof(T) == sizeof(U)>>
const_array_ref<U, Shape> reinterpret(const array<T, Shape, Alloc>& a) {
  return reinterpret<const U>(a.cref());
}

/** Reinterpret the shape of the array or array_ref `a` to be a new shape
 * `new_shape`, with a base pointer offset `offset`. */
template <class NewShape, class T, class OldShape>
NDARRAY_HOST_DEVICE array_ref<T, NewShape> reinterpret_shape(
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
NDARRAY_HOST_DEVICE auto transpose(const array_ref<T, OldShape>& a) {
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
NDARRAY_HOST_DEVICE auto reorder(const array_ref<T, OldShape>& a) {
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

/** Allocator satisfying the `std::allocator` interface that owns a buffer with
 * automatic storage, and a fallback base allocator. For allocations, the
 * allocator uses the buffer if it is large enough and not already allocated,
 * otherwise it uses the base allocator. */
template <class T, size_t N, size_t Alignment = alignof(T), class BaseAlloc = std::allocator<T>>
class auto_allocator {
  alignas(Alignment) char buffer[N * sizeof(T)];
  bool allocated;
  BaseAlloc alloc;

public:
  using value_type = T;

  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::false_type;

  static auto_allocator select_on_container_copy_construction(const auto_allocator&) {
    return auto_allocator();
  }

  auto_allocator() : allocated(false) {}
  template <class U, size_t U_N, size_t U_A, class U_BaseAlloc>
  constexpr auto_allocator(const auto_allocator<U, U_N, U_A, U_BaseAlloc>&) noexcept
      : allocated(false) {}
  // These constructors/assignment operators are workarounds for a C++
  // STL implementation not respecting the propagate typedefs or the
  // 'select_on_...' function. (https://github.com/dsharlet/array/issues/7)
  auto_allocator(const auto_allocator& copy) noexcept : allocated(false), alloc(copy.alloc) {}
  auto_allocator(auto_allocator&& move) noexcept : allocated(false), alloc(std::move(move.alloc)) {}
  auto_allocator& operator=(const auto_allocator& copy) {
    alloc = copy.alloc;
    return *this;
  }
  auto_allocator& operator=(auto_allocator&& move) {
    alloc = std::move(move.alloc);
    return *this;
  }

  value_type* allocate(size_t n) {
    if (!allocated && n <= N) {
      allocated = true;
      return reinterpret_cast<value_type*>(&buffer[0]);
    } else {
      return std::allocator_traits<BaseAlloc>::allocate(alloc, n);
    }
  }
  void deallocate(value_type* ptr, size_t n) noexcept {
    if (ptr == reinterpret_cast<value_type*>(&buffer[0])) {
      assert(allocated);
      allocated = false;
    } else {
      std::allocator_traits<BaseAlloc>::deallocate(alloc, ptr, n);
    }
  }

  template <class U, size_t U_N, size_t U_A>
  friend bool operator==(const auto_allocator& a, const auto_allocator<U, U_N, U_A>& b) {
    if (a.allocated || b.allocated) {
      return &a.buffer[0] == &b.buffer[0];
    } else {
      return a.alloc == b.alloc;
    }
  }

  template <class U, size_t U_N, size_t U_A>
  friend bool operator!=(const auto_allocator& a, const auto_allocator<U, U_N, U_A>& b) {
    return !(a == b);
  }
};

/** Allocator satisfying the `std::allocator` interface that is a wrapper
 * around another allocator `BaseAlloc`, and skips default construction.
 * Using this allocator can be dangerous. It is only safe to use when
 * `BaseAlloc::value_type` is a trivial type. */
template <class BaseAlloc>
class uninitialized_allocator : public BaseAlloc {
public:
  using value_type = typename std::allocator_traits<BaseAlloc>::value_type;

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

  value_type* allocate(size_t n) { return std::allocator_traits<BaseAlloc>::allocate(*this, n); }
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

  template <class OtherBaseAlloc>
  friend bool operator==(
      const uninitialized_allocator& a, const uninitialized_allocator<OtherBaseAlloc>& b) {
    return static_cast<const BaseAlloc&>(a) == static_cast<const OtherBaseAlloc&>(b);
  }
  template <class OtherBaseAlloc>
  friend bool operator!=(
      const uninitialized_allocator& a, const uninitialized_allocator<OtherBaseAlloc>& b) {
    return static_cast<const BaseAlloc&>(a) != static_cast<const OtherBaseAlloc&>(b);
  }
};

/** Allocator equivalent to `std::allocator<T>` that does not default
 * construct values. */
template <class T, class = std::enable_if_t<std::is_trivial<T>::value>>
using uninitialized_std_allocator = uninitialized_allocator<std::allocator<T>>;

/** Allocator equivalent to `auto_allocator<T, N, Alignment>` that
 * does not default construct values. */
template <class T, size_t N, size_t Alignment = sizeof(T),
    class = std::enable_if_t<std::is_trivial<T>::value>>
using uninitialized_auto_allocator = uninitialized_allocator<auto_allocator<T, N, Alignment>>;

} // namespace nda

#endif // NDARRAY_ARRAY_H

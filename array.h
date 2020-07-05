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
#include <functional>
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
 * optimize loops. */
#ifdef NDARRAY_INT_INDICES
using index_t = int;
#else
using index_t = std::ptrdiff_t;
#endif

/** This value indicates a compile-time constant stride is unknown, and to use
 * the corresponding runtime value instead. */
// It would be better to use a more unreasonable value that would never be
// used in practice. Fortunately, this does not affect correctness, only
// performance, and it is hard to imagine a use case for this where
// performance matters.
constexpr index_t UNK = -9;

namespace internal {

NDARRAY_INLINE index_t abs(index_t a) { return a >= 0 ? a : -a; }

NDARRAY_INLINE constexpr index_t is_known(index_t x) { return x != UNK; }
NDARRAY_INLINE constexpr index_t is_unknown(index_t x) { return x == UNK; }

// Given a compile-time static value, reconcile a compile-time static value and
// runtime value.
template <index_t Value>
NDARRAY_INLINE constexpr index_t reconcile(index_t value) {
  // It would be nice to assert here that Value == value. But, this is used in
  // the innermost loops, so when asserts are on, this ruins performance. It
  // is also a less helpful place to catch errors, because the context of the
  // bug is lost here.
  return is_known(Value) ? Value : value;
}

constexpr bool is_unknown(index_t a, index_t b) { return is_unknown(a) || is_unknown(b); }

template <index_t A, index_t B>
using enable_if_compatible = typename std::enable_if<is_unknown(A, B) || A == B>::type;

constexpr index_t add(index_t a, index_t b) { return is_unknown(a, b) ? UNK : a + b; }
constexpr index_t sub(index_t a, index_t b) { return is_unknown(a, b) ? UNK : a - b; }
constexpr index_t mul(index_t a, index_t b) { return is_unknown(a, b) ? UNK : a * b; }
constexpr index_t min(index_t a, index_t b) { return is_unknown(a, b) ? UNK : (a < b ? a : b); }
constexpr index_t max(index_t a, index_t b) { return is_unknown(a, b) ? UNK : (a > b ? a : b); }

}  // namespace internal

/** An iterator over a range of indices, enabling range-based for loops for
 * indices. */
class index_iterator {
  index_t i_;

 public:
  index_iterator(index_t i) : i_(i) {}

  NDARRAY_INLINE bool operator==(const index_iterator& r) const { return i_ == r.i_; }
  NDARRAY_INLINE bool operator!=(const index_iterator& r) const { return i_ != r.i_; }

  NDARRAY_INLINE index_t operator *() const { return i_; }

  NDARRAY_INLINE index_iterator operator++(int) { return index_iterator(i_++); }
  NDARRAY_INLINE index_iterator& operator++() { ++i_; return *this; }
};

/** Describes a range of indices. The template parameters enable providing
 * compile time constants for the `min` and `extent` of the range. These
 * parameters Values not in the range [min, min + extent) are considered to be
 * out of bounds. */
template <index_t Min_ = UNK, index_t Extent_ = UNK>
class range {
 protected:
  index_t min_;
  index_t extent_;

 public:
  static constexpr index_t Min = Min_;
  static constexpr index_t Extent = Extent_;
  static constexpr index_t Max = internal::sub(internal::add(Min, Extent), 1);

  /** Construct a new range object. If the class template parameters `Min`
   * or `Extent` are not `UNK`, these runtime values must match the
   * compile-time values. */
  range(index_t min = Min, index_t extent = Extent) {
    set_min(min);
    set_extent(extent);
  }
  range(const range&) = default;
  range(range&&) = default;
  /** Copy another range object, possibly with different compile-time template
   * parameters. */
  template <index_t CopyMin, index_t CopyExtent,
      class = internal::enable_if_compatible<Min, CopyMin>,
      class = internal::enable_if_compatible<Extent, CopyExtent>>
  range(const range<CopyMin, CopyExtent>& other) : range(other.min(), other.extent()) {}

  range& operator=(const range&) = default;
  range& operator=(range&&) = default;
  /** Copy assignment of a range object, possibly with different compile-time
   * template parameters. */
  template <index_t CopyMin, index_t CopyExtent,
      class = internal::enable_if_compatible<Min, CopyMin>,
      class = internal::enable_if_compatible<Extent, CopyExtent>>
  range& operator=(const range<CopyMin, CopyExtent>& other) {
    set_min(other.min());
    set_extent(other.extent());
    return *this;
  }

  /** Index of the first element in this range. */
  NDARRAY_INLINE index_t min() const { return internal::reconcile<Min>(min_); }
  NDARRAY_INLINE void set_min(index_t min) {
    if (internal::is_unknown(Min)) {
      min_ = min;
    } else {
      assert(min == Min);
    }
  }
  /** Number of elements in this range. */
  NDARRAY_INLINE index_t extent() const { return internal::reconcile<Extent>(extent_); }
  NDARRAY_INLINE void set_extent(index_t extent) {
    if (internal::is_unknown(Extent)) {
      extent_ = extent;
    } else {
      assert(extent == Extent);
    }
  }
  /** Index of the last element in this range. */
  NDARRAY_INLINE index_t max() const { return min() + extent() - 1; }
  NDARRAY_INLINE void set_max(index_t max) { set_extent(max - min() + 1); }

  /** Returns true if `at` is within the range [`min()`, `max()`]. */
  NDARRAY_INLINE bool is_in_range(index_t at) const { return min() <= at && at <= max(); }
  template <index_t OtherMin, index_t OtherExtent>
  NDARRAY_INLINE bool is_in_range(const range<OtherMin, OtherExtent>& at) const {
    return min() <= at.min() && at.max() <= max();
  }

  /** Make an iterator referring to the first element in this range. */
  index_iterator begin() const { return index_iterator(min()); }
  /** Make an iterator referring to one past the last element in this range. */
  index_iterator end() const { return index_iterator(max() + 1); }

  /** Two range objects are considered equal if their mins and extents
   * are equal. */
  template <index_t OtherMin, index_t OtherExtent,
      class = internal::enable_if_compatible<Min, OtherMin>,
      class = internal::enable_if_compatible<Extent, OtherExtent>>
  bool operator==(const range<OtherMin, OtherExtent>& other) const {
    return min() == other.min() && extent() == other.extent();
  }
  template <index_t OtherMin, index_t OtherExtent,
      class = internal::enable_if_compatible<Min, OtherMin>,
      class = internal::enable_if_compatible<Extent, OtherExtent>>
  bool operator!=(const range<OtherMin, OtherExtent>& other) const {
    return !operator==(other);
  }
};

/** Specialization of `range` where the min is unknown. */
template <index_t Extent>
using fixed_range = range<UNK, Extent>;

/** Make a range from a half-open interval [`begin`, `end`). */
inline range<> interval(index_t begin, index_t end) {
  return range<>(begin, end - begin);
}

/** A `range` that means the entire dimension. */
const range<0, -1> _;

/** Overloads of `std::begin` and `std::end` */
template <index_t Min, index_t Extent>
index_iterator begin(const range<Min, Extent>& d) { return d.begin(); }
template <index_t Min, index_t Extent>
index_iterator end(const range<Min, Extent>& d) { return d.end(); }

/** Clamp an index to the range [min, max]. */
inline index_t clamp(index_t x, index_t min, index_t max) {
  return std::min(std::max(x, min), max);
}

/** Clamp an index to the range described by an object with a min and max method. */
template <class Dim>
index_t clamp(index_t x, const Dim& d) {
  return clamp(x, d.min(), d.max());
}

namespace internal {

// An iterator for a range of ranges.
template <index_t InnerExtent = UNK>
class split_iterator {
  fixed_range<InnerExtent> i;
  index_t outer_max;

 public:
  split_iterator(const fixed_range<InnerExtent>& i, index_t outer_max)
      : i(i), outer_max(outer_max) {}

  bool operator==(const split_iterator& r) const { return i.min() == r.i.min(); }
  bool operator!=(const split_iterator& r) const { return i.min() != r.i.min(); }

  fixed_range<InnerExtent> operator *() const { return i; }

  split_iterator& operator++() {
    if (is_known(InnerExtent)) {
      // When the extent of the inner split is a compile-time constant,
      // we can't shrink the out of bounds range. Instead, shift the min,
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

template <index_t InnerExtent = UNK>
using split_iterator_range = iterator_range<split_iterator<InnerExtent>>;

}  // namespace internal

/** Split a range `r` into an iteratable range of ranges by a compile-time
 * constant `InnerExtent`. If `InnerExtent` does not divide `r.extent()`,
 * the last range will be shifted to overlap with the second-to-last iteration,
 * to preserve the compile-time constant extent, which implies `r.extent()`
 * must be larger `InnerExtent`. */
template <index_t InnerExtent, index_t Min, index_t Extent>
internal::split_iterator_range<InnerExtent> split(const range<Min, Extent>& r) {
  assert(r.extent() >= InnerExtent);
  return {
      {fixed_range<InnerExtent>(r.min()), r.max()},
      {fixed_range<InnerExtent>(r.max() + 1), r.max()}};
}

/** Split a range `r` into an iterable range of ranges by `inner_extent`. If
 * `InnerExtent` does not divide `r.extent()`, the last iteration will be
 * clamped to the outer range. */
// TODO: This probably doesn't need to be templated, but it might help
// avoid some conversion messes. dim<Min, Extent> probably can't implicitly
// convert to range<>.
template <index_t Min, index_t Extent>
internal::split_iterator_range<> split(const range<Min, Extent>& r, index_t inner_extent) {
  return {
      {range<>(r.min(), inner_extent), r.max()},
      {range<>(r.max() + 1, inner_extent), r.max()}};
}

/** Describes one dimension of an array. The template parameters enable
 * providing compile time constants for the `min`, `extent`, and `stride` of the
 * dim. These parameters define a mapping from the indices of the dimension to
 * offsets: offset(x) = (x - min)*stride. The extent does not affect the mapping
 * directly. Values not in the range [min, min + extent) are considered to be
 * out of bounds. */
// TODO: Consider adding helper class constant<Value> to use for the members of
// dim. (https://github.com/dsharlet/array/issues/1)
template <index_t Min_ = UNK, index_t Extent_ = UNK, index_t Stride_ = UNK>
class dim : public range<Min_, Extent_> {
 protected:
  index_t stride_;

 public:
  using base_range = range<Min_, Extent_>;

  using base_range::Min;
  using base_range::Extent;
  using base_range::Max;

  static constexpr index_t Stride = Stride_;

  /** Construct a new dim object. If the class template parameters `Min`,
   * `Extent`, or `Stride` are not `UNK`, these runtime values must match the
   * compile-time values. */
  dim(index_t min, index_t extent, index_t stride = Stride) : base_range(min, extent) {
    set_stride(stride);
  }
  dim(index_t extent) : dim(0, extent) {}
  dim() : dim(Min, Extent, Stride) {}
  dim(const base_range& range, index_t stride = Stride)
      : dim(range.min(), range.extent(), stride) {}
  dim(const dim&) = default;
  dim(dim&&) = default;
  /** Copy another dim object, possibly with different compile-time template
   * parameters. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride,
      class = internal::enable_if_compatible<Min, CopyMin>,
      class = internal::enable_if_compatible<Extent, CopyExtent>,
      class = internal::enable_if_compatible<Stride, CopyStride>>
  dim(const dim<CopyMin, CopyExtent, CopyStride>& other)
      : dim(other.min(), other.extent(), other.stride()) {}

  dim& operator=(const dim&) = default;
  dim& operator=(dim&&) = default;
  /** Copy assignment of a dim object, possibly with different compile-time
   * template parameters. */
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

  /** Distance in flat indices between neighboring elements in this dim. */
  NDARRAY_INLINE index_t stride() const { return internal::reconcile<Stride>(stride_); }
  NDARRAY_INLINE void set_stride(index_t stride) {
    if (internal::is_unknown(Stride)) {
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

/** Specialization of `dim` where the min is not specified at compile time. */
template <index_t Extent, index_t Stride = UNK>
using fixed_dim = dim<UNK, Extent, Stride>;

/** Specialization of `dim` where the compile-time stride parameter is known
 * to be one. */
template <index_t Min = UNK, index_t Extent = UNK>
using dense_dim = dim<Min, Extent, 1>;

/** Specialization of `dim` where only the stride parameter is specified at
 * compile time. */
template <index_t Stride>
using strided_dim = dim<UNK, UNK, Stride>;

/** Specialization of `dim` where the compile-time stride parameter is known
 * to be zero. */
template <index_t Min = UNK, index_t Extent = UNK>
using broadcast_dim = dim<Min, Extent, 0>;

namespace internal {

using std::index_sequence;
using std::make_index_sequence;

// Call `fn` with the elements of tuple `args` unwrapped from the tuple.
// TODO: When we assume C++17, this can be replaced by std::apply.
template <class Fn, class Args, size_t... Is>
NDARRAY_INLINE auto apply(Fn&& fn, const Args& args, index_sequence<Is...>) {
  return fn(std::get<Is>(args)...);
}

template <class Fn, class... Args>
NDARRAY_INLINE auto apply(Fn&& fn, const std::tuple<Args...>& args) {
  return apply(fn, args, make_index_sequence<sizeof...(Args)>());
}

// Some variadic reduction helpers.
NDARRAY_INLINE index_t sum() { return 0; }
template <class... Rest>
NDARRAY_INLINE index_t sum(index_t first, Rest... rest) {
  return first + sum(rest...);
}

NDARRAY_INLINE index_t product() { return 1; }
template <class... Rest>
NDARRAY_INLINE index_t product(index_t first, Rest... rest) {
  return first * product(rest...);
}

NDARRAY_INLINE index_t variadic_min() { return std::numeric_limits<index_t>::max(); }
template <class... Rest>
NDARRAY_INLINE index_t variadic_min(index_t first, Rest... rest) {
  return std::min(first, variadic_min(rest...));
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
index_t flat_offset_pack(const Dims& dims) {
  return 0;
}

template <size_t D, class Dims, class... Indices>
index_t flat_offset_pack(const Dims& dims, index_t i0, Indices... indices) {
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

template <index_t DimMin, index_t DimExtent, index_t DimStride>
auto range_with_stride(index_t x, const dim<DimMin, DimExtent, DimStride>& d) {
  return dim<UNK, 1, DimStride>(x, 1, d.stride());
}

template <index_t CropMin, index_t CropExtent, index_t DimMin, index_t DimExtent, index_t Stride>
auto range_with_stride(
    const range<CropMin, CropExtent>& x, const dim<DimMin, DimExtent, Stride>& d) {
  return dim<CropMin, CropExtent, Stride>(x.min(), x.extent(), d.stride());
}

template <index_t Min, index_t Extent, index_t Stride>
auto range_with_stride(const decltype(_)&, const dim<Min, Extent, Stride>& d) {
  return d;
}

template <class Ranges, class Dims, size_t... Is>
auto ranges_with_strides(const Ranges& ranges, const Dims& dims, index_sequence<Is...>) {
  return std::make_tuple(range_with_stride(std::get<Is>(ranges), std::get<Is>(dims))...);
}

// Make a tuple of dims corresponding to elements in ranges that are not slices.
template <class Dim>
std::tuple<> skip_slices_impl(const Dim& dim, index_t) {
  return std::tuple<>();
}

template <class Dim>
std::tuple<Dim> skip_slices_impl(const Dim& dim, const range<>&) {
  return std::tuple<Dim>(dim);
}

template <class Dims, class Ranges, size_t... Is>
auto skip_slices(const Dims& dims, const Ranges& ranges, index_sequence<Is...>) {
  return std::tuple_cat(skip_slices_impl(std::get<Is>(dims), std::get<Is>(ranges))...);
}

// Checks if all indices are in range of each corresponding dim.
template <class Dims, class Indices, size_t... Is>
bool is_in_range(const Dims& dims, const Indices& indices, index_sequence<Is...>) {
  return all(std::get<Is>(dims).is_in_range(std::get<Is>(indices))...);
}

// We want to be able to call mins on a mixed tuple of int/index_t, range, and dim.
template <class Dim>
index_t min_of_range(index_t x, const Dim&) {
  return x;
}

template <index_t Min, index_t Extent, class Dim>
index_t min_of_range(const range<Min, Extent>& x, const Dim&) {
  return x.min();
}

template <class Dim>
index_t min_of_range(const decltype(_)&, const Dim& dim) {
  return dim.min();
}

template <class Ranges, class Dims, size_t... Is>
auto mins_of_ranges(const Ranges& ranges, const Dims& dims, index_sequence<Is...>) {
  return std::make_tuple(min_of_range(std::get<Is>(ranges), std::get<Is>(dims))...);
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

template <class Dim>
bool is_dim_ok(index_t stride, index_t extent, const Dim& dim) {
  if (is_unknown(dim.stride())) {
    // If the dimension has an unknown stride, it's OK, we're resolving the
    // current dim first.
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
bool is_dim_ok(index_t stride, index_t extent, const AllDims& all_dims, index_sequence<Is...>) {
  return all(is_dim_ok(stride, extent, std::get<Is>(all_dims))...);
}

template <class AllDims>
index_t filter_stride(index_t stride, index_t extent, const AllDims& all_dims) {
  constexpr size_t rank = std::tuple_size<AllDims>::value;
  if (is_dim_ok(stride, extent, all_dims, make_index_sequence<rank>())) {
    return stride;
  } else {
    return std::numeric_limits<index_t>::max();
  }
}

template <size_t D, class AllDims>
index_t candidate_stride(index_t extent, const AllDims& all_dims) {
  index_t stride_d = std::get<D>(all_dims).stride();
  if (is_unknown(stride_d)) {
    return std::numeric_limits<index_t>::max();
  }
  index_t stride =
      std::max(static_cast<index_t>(1), abs(stride_d) * std::get<D>(all_dims).extent());
  return filter_stride(stride, extent, all_dims);
}

template <class AllDims, size_t... Is>
index_t find_stride(index_t extent, const AllDims& all_dims, index_sequence<Is...>) {
  return variadic_min(
      filter_stride(1, extent, all_dims), candidate_stride<Is>(extent, all_dims)...);
}

inline void resolve_unknown_extents() {}

template <class Dim0, class... Dims>
void resolve_unknown_extents(Dim0& dim0, Dims&... dims) {
  if (is_unknown(dim0.extent())) {
    dim0.set_extent(0);
  }
  resolve_unknown_extents(dims...);
}

template <class Dims, size_t... Is>
void resolve_unknown_extents(Dims& dims, index_sequence<Is...>) {
  resolve_unknown_extents(std::get<Is>(dims)...);
}


template <class AllDims>
void resolve_unknown_strides(AllDims& all_dims) {}

template <class AllDims, class Dim0, class... Dims>
void resolve_unknown_strides(AllDims& all_dims, Dim0& dim0, Dims&... dims) {
  if (is_unknown(dim0.stride())) {
    constexpr size_t rank = std::tuple_size<AllDims>::value;
    dim0.set_stride(find_stride(dim0.extent(), all_dims, make_index_sequence<rank>()));
  }
  resolve_unknown_strides(all_dims, dims...);
}

template <class Dims, size_t... Is>
void resolve_unknown_strides(Dims& dims, index_sequence<Is...>) {
  resolve_unknown_strides(dims, std::get<Is>(dims)...);
}

template <class Dims>
void resolve_unknowns(Dims& dims) {
  constexpr size_t rank = std::tuple_size<Dims>::value;
  resolve_unknown_extents(dims, make_index_sequence<rank>());
  resolve_unknown_strides(dims, make_index_sequence<rank>());
}

template <class Dim>
bool is_known(const Dim& dim) {
  return is_known(dim.min()) && is_known(dim.extent()) && is_known(dim.stride());
}

template<class Dims, size_t... Is>
bool all_known(const Dims& dims, index_sequence<Is...>) {
  return all(is_known(std::get<Is>(dims))...);
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

}  // namespace internal

template <class... Dims>
class shape;

/** Helper function to make a tuple from a variadic list of dims. */
template <class... Dims>
auto make_shape(Dims... dims) {
  return shape<Dims...>(dims...);
}

template <class... Dims>
shape<Dims...> make_shape_from_tuple(const std::tuple<Dims...>& dims) {
  return shape<Dims...>(dims);
}

/** A list of `dim` objects describing a multi-dimensional space of indices.
 * The `rank` of a shape refers to the number of dimensions in the shape.
 * Shapes map multiple dim objects to offsets by adding each mapping dim to
 * offset together to produce a 'flat offset'. The first dimension is known as
 * the 'innermost' dimension, and dimensions then increase until the
 * 'outermost' dimension. */
template <class... Dims>
class shape {
  std::tuple<Dims...> dims_;

 public:
  /** Number of dims in this shape. */
  static constexpr size_t rank() { return std::tuple_size<std::tuple<Dims...>>::value; }

  /** A shape is scalar if it is rank 0. */
  static constexpr bool is_scalar() { return rank() == 0; }

  /** The type of an index for this shape. */
  using index_type = typename internal::tuple_of_n<index_t, rank()>::type;

  using size_type = size_t;

 private:
  // TODO: This should use std::is_constructible<std::tuple<Dims...>, std::tuple<OtherDims...>>
  // but it is broken on some compilers (https://github.com/dsharlet/array/issues/20).
  template <class... OtherDims>
  using enable_if_dims_compatible = typename std::enable_if<sizeof...(OtherDims) == rank()>::type;

  template <class... Args>
  using enable_if_same_rank = typename std::enable_if<sizeof...(Args) == rank()>::type;

  template <class... Args>
  using enable_if_indices =
      typename std::enable_if<internal::all_of_type<index_t, Args...>::value>::type;

  template <class... Args>
  using enable_if_ranges = typename std::enable_if<
      internal::all_of_type<range<>, Args...>::value &&
      !internal::all_of_type<index_t, Args...>::value>::type;

  template <size_t Dim>
  using enable_if_dim = typename std::enable_if<Dim < rank()>::type;

 public:
  /** When constructing shapes, unknown extents are set to 0, and unknown
   * strides are set to the currently largest known stride. This is done in
   * innermost-to-outermost order. */
  // TODO: Don't resolve unknowns upon shape construction, do it only when
  // constructing arrays (and not array_refs).
  shape() { resolve(); }
  // TODO: This is a bit messy, but necessary to avoid ambiguous default
  // constructors when Dims is empty.
  template <size_t N = sizeof...(Dims), class = typename std::enable_if<(N > 0)>::type>
  shape(const Dims&... dims) : dims_(dims...) { resolve(); }
  shape(const shape&) = default;
  shape(shape&&) = default;

  /** Construct a shape from a tuple of `dims` of another type. */
  // We cannot have an std::tuple<Dims...> constructor because it will be
  // ambiguous with the Dims... constructor for 1D shapes.
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape(const std::tuple<OtherDims...>& dims) : dims_(dims) { resolve(); }
  /** Construct a shape from a different type of `dims`. */
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape(OtherDims... dims) : dims_(dims...) { resolve(); }

  /** Construct this shape from a different type of shape. `conversion` must
   * be convertible to this shape. */
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape(const shape<OtherDims...>& conversion) : dims_(conversion.dims()) { resolve(); }

  shape& operator=(const shape&) = default;
  shape& operator=(shape&&) = default;

  /** Assign this shape from a different type of shape. `conversion` must be
   * convertible to this shape. */
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  shape& operator=(const shape<OtherDims...>& conversion) {
    dims_ = conversion.dims();
    return *this;
  }

  /* When constructing arrays, unknown extents are set to 0, and unknown
   * strides are set to the currently largest known stride. This is done in
   * innermost-to-outermost order. */
  void resolve() { internal::resolve_unknowns(dims_); }

  /** Check if all values of the shape are known. */
  bool is_known() const {
    return internal::all_known(dims_, internal::make_index_sequence<rank()>());
  }

  /** Returns true if the index `indices` are in range of this shape. */
  bool is_in_range(const index_type& indices) const {
    return internal::is_in_range(dims_, indices, internal::make_index_sequence<rank()>());
  }
  // This supports both indices and ranges. It appears not to be possible
  // to have two overloads differentiated by enable_if_indices and
  // enable_if_ranges.
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
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_indices<Args...>>
  index_t operator() (Args... indices) const {
    return internal::flat_offset_pack<0>(dims_, indices...);
  }

  /** Create a new shape using the specified crops and slices in `args`.
   * The resulting shape will have the sliced dimensions removed. */
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_ranges<Args...>>
  auto operator() (Args... args) const {
    auto ranges_tuple = std::make_tuple(args...);
    auto new_dims =
        internal::ranges_with_strides(ranges_tuple, dims_, internal::make_index_sequence<rank()>());
    auto new_dims_no_slices =
        internal::skip_slices(new_dims, ranges_tuple, internal::make_index_sequence<rank()>());
    return make_shape_from_tuple(new_dims_no_slices);
  }

  /** Get a specific dim of this shape. */
  template <size_t D, class = enable_if_dim<D>>
  auto& dim() { return std::get<D>(dims_); }
  template <size_t D, class = enable_if_dim<D>>
  const auto& dim() const { return std::get<D>(dims_); }
  /** Get a specific dim of this shape with a runtime dimension d. This will
   * lose knowledge of any compile-time constant dimension attributes. */
  nda::dim<> dim(size_t d) const {
    assert(d < rank());
    return internal::tuple_to_array<nda::dim<>>(dims_)[d];
  }

  /** Get a tuple of all of the dims of this shape. */
  std::tuple<Dims...>& dims() { return dims_; }
  const std::tuple<Dims...>& dims() const { return dims_; }

  /** Get an index pointing to the min or max index in each dimension of this
   * shape. */
  index_type min() const { return internal::mins(dims(), internal::make_index_sequence<rank()>()); }
  index_type max() const { return internal::maxs(dims(), internal::make_index_sequence<rank()>()); }
  index_type extent() const {
    return internal::extents(dims(), internal::make_index_sequence<rank()>());
  }
  index_type stride() const {
    return internal::strides(dims(), internal::make_index_sequence<rank()>());
  }

  /** Compute the flat extent of this shape. This is the extent of the valid
   * range of values returned by at or operator(). */
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

  /** Compute the total number of items in the shape. */
  size_type size() const {
    index_t s = internal::product(extent(), internal::make_index_sequence<rank()>());
    return s < 0 ? 0 : static_cast<size_type>(s);
  }

  /** A shape is empty if its size is 0. */
  bool empty() const { return size() == 0; }

  /** Returns true if this shape is `compact` in memory. A shape is compact if
   * there are no unaddressable flat indices between the first and last
   * addressable flat elements. */
  bool is_compact() const { return flat_extent() <= size(); }

  /** Returns true if this shape is an injective function mapping indices to
   * flat indices. If the dims overlap, or a dim has stride zero, multiple
   * indices will map to the same flat index; in this case, this function will
   * return false. */
  bool is_one_to_one() const {
    // TODO: https://github.com/dsharlet/array/issues/2
    return flat_extent() >= size();
  }

  /** Returns true if this shape projects to a set of flat indices that is a
   * subset of the other shape's projection to flat indices with an offset. */
  template <typename OtherShape>
  bool is_subset_of(const OtherShape& other, index_t offset) const {
    // TODO: https://github.com/dsharlet/array/issues/2
    return flat_min() >= other.flat_min() + offset && flat_max() <= other.flat_max() + offset;
  }

  /** Provide some aliases for common interpretations of dimensions. */
  auto& i() { return dim<0>(); }
  const auto& i() const { return dim<0>(); }
  auto& j() { return dim<1>(); }
  const auto& j() const { return dim<1>(); }
  auto& k() { return dim<2>(); }
  const auto& k() const { return dim<2>(); }

  auto& x() { return dim<0>(); }
  const auto& x() const { return dim<0>(); }
  auto& y() { return dim<1>(); }
  const auto& y() const { return dim<1>(); }
  auto& z() { return dim<2>(); }
  const auto& z() const { return dim<2>(); }
  auto& w() { return dim<3>(); }
  const auto& w() const { return dim<3>(); }

  auto& c() { return dim<2>(); }
  const auto& c() const { return dim<2>(); }

  /** Assuming this array represents an image with dimensions {width, height,
   * channels}, get the extent of those dimensions. */
  index_t width() const { return x().extent(); }
  index_t height() const { return y().extent(); }
  index_t channels() const { return c().extent(); }

  /** Assuming this array represents a matrix with dimensions {rows,
   * cols}, get the extent of those dimensions. */
  index_t rows() const { return i().extent(); }
  index_t columns() const { return j().extent(); }

  /** A shape is equal to another shape if the dim objects of
   * each dimension from both shapes are equal. */
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  bool operator==(const shape<OtherDims...>& other) const { return dims_ == other.dims(); }
  template <class... OtherDims, class = enable_if_dims_compatible<OtherDims...>>
  bool operator!=(const shape<OtherDims...>& other) const { return dims_ != other.dims(); }
};

/** Create a new shape using a list of DimIndices to use as the dimensions of
 * the shape. */
template <size_t... DimIndices, class Shape>
auto select_dims(const Shape& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}
template <size_t... DimIndices, class Shape>
auto reorder(const Shape& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}

namespace internal {

template <class Fn, class... Ts>
using enable_if_callable = decltype(std::declval<Fn>()(std::declval<Ts>()...));

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

template <size_t Rank, size_t... Is>
auto make_default_dense_shape() {
  return make_shape_from_tuple(std::tuple_cat(
      std::make_tuple(dense_dim<>()), typename tuple_of_n<dim<>, Rank - 1>::type()));
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
    (is_unknown(Min) || src.min() == Min) &&
    (is_unknown(Extent) || src.extent() == Extent) &&
    (is_unknown(Stride) || src.stride() == Stride);
}

template <class... DimsDst, class ShapeSrc, size_t... Is>
bool is_shape_compatible(const shape<DimsDst...>&, const ShapeSrc& src, index_sequence<Is...>) {
  return all(is_dim_compatible(DimsDst(), src.template dim<Is>())...);
}

template <class DimA, class DimB>
auto intersect_dims(const DimA& a, const DimB& b) {
  constexpr index_t Min = max(DimA::Min, DimB::Min);
  constexpr index_t Max = min(DimA::Max, DimB::Max);
  constexpr index_t Extent = add(sub(Max, Min), 1);
  constexpr index_t Stride = DimA::Stride == DimB::Stride ? DimA::Stride : UNK;
  index_t min = std::max(a.min(), b.min());
  index_t max = std::min(a.max(), b.max());
  index_t extent = max - min + 1;
  return dim<Min, Extent, Stride>(min, extent);
}

template <class DimsA, class DimsB, size_t... Is>
auto intersect(const DimsA& a, const DimsB& b, index_sequence<Is...>) {
  return make_shape(intersect_dims(std::get<Is>(a), std::get<Is>(b))...);
}

}  // namespace internal

/** Make a shape with an equivalent domain of indices, with dense strides. */
template <class... Dims>
auto make_dense(const shape<Dims...>& shape) {
  constexpr size_t rank = sizeof...(Dims);
  return internal::make_dense_shape(shape.dims(), internal::make_index_sequence<rank - 1>());
}

/** Make a shape with an equivalent domain of indices, but with compact strides.
 * Only required strides are respected. */
template <class Shape>
Shape make_compact(const Shape& s) {
  Shape without_strides =
      internal::without_strides(s, internal::make_index_sequence<Shape::rank()>());
  without_strides.resolve();
  return without_strides;
}

/** An arbitrary shape (no compile-time constant parameters) with the specified
 * Rank. */
template <size_t Rank>
using shape_of_rank =
    decltype(make_shape_from_tuple(typename internal::tuple_of_n<dim<>, Rank>::type()));

/** A shape where the innermost dimension is a `dense_dim` object, and all other
 * dimensions are arbitrary. */
template <size_t Rank>
using dense_shape = decltype(internal::make_default_dense_shape<Rank>());

/** Test if a shape `src` can be assigned to a shape of type `ShapeDst` without
 * error. */
// Unfortunately, this is backwards from std::is_convertible. But the other way
// around doesn't work without forcing the caller to specify ShapeSrc when it
// should be inferred.
template <class ShapeDst, class ShapeSrc>
bool is_compatible(const ShapeSrc& src) {
  static_assert(ShapeSrc::rank() == ShapeDst::rank(), "shapes must have the same rank.");
  return internal::is_shape_compatible(
      ShapeDst(), src, internal::make_index_sequence<ShapeSrc::rank()>());
}

/** Compute the intersection of two shapes `a` and `b`. The intersection is the
 * shape containing the indices in bounds of both `a` and `b`. */
template <class ShapeA, class ShapeB>
auto intersect(const ShapeA& a, const ShapeB& b) {
  constexpr size_t rank = ShapeA::rank() < ShapeB::rank() ? ShapeA::rank() : ShapeB::rank();
  return internal::intersect(a.dims(), b.dims(), internal::make_index_sequence<rank>());
}

/** Iterate over all indices in the shape, calling a function `fn` for each set
 * of indices. The indices are in the same order as the dims in the shape. The
 * first dim is the `inner` loop of the iteration, and the last dim is the
 * `outer` loop.
 *
 * These functions are typically used to implement shape_traits and
 * copy_shape_traits objects. Use for_each_index, array_ref<>::for_each_value,
 * or array<>::for_each_value instead. */
template<class Shape, class Fn,
    class = internal::enable_if_callable<Fn, typename Shape::index_type>>
void for_each_index_in_order(const Shape& shape, Fn &&fn) {
  internal::for_each_index_in_order<Shape::rank() - 1>(shape.dims(), fn, std::tuple<>());
}
template<class Shape, class Ptr, class Fn>
void for_each_value_in_order(const Shape& shape, Ptr base, Fn &&fn) {
  using index_type = typename Shape::index_type;
  // TODO: This is losing compile-time constant extents and strides info
  // (https://github.com/dsharlet/array/issues/1).
  std::tuple<Ptr, index_type> base_and_stride(base, shape.stride());
  internal::for_each_value_in_order<Shape::rank() - 1>(shape.extent(), fn, base_and_stride);
}

/** Similar to for_each_value_in_order, but iterates over two arrays
 * simultaneously. `shape` defines the loop nest, while `shape_a` and `shape_b`
 * define the memory layout of `base_a` and `base_b`. */
template<class Shape, class ShapeA, class PtrA, class ShapeB, class PtrB, class Fn>
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

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_compatible =
    typename std::enable_if<std::is_constructible<ShapeDst, ShapeSrc>::value>::type;

template <class ShapeDst, class ShapeSrc>
using enable_if_shapes_copy_compatible =
    typename std::enable_if<ShapeDst::rank() == ShapeSrc::rank()>::type;

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

/** Shape traits enable some behaviors to be overriden per shape type. */
template <class Shape>
class shape_traits {
 public:
  using shape_type = Shape;

  /** The for_each_index implementation for the shape may choose to iterate in a
   * different order than the default (in-order). */
  template <class Fn>
  static void for_each_index(const Shape& shape, Fn&& fn) {
    for_each_index_in_order(shape, fn);
  }

  /** The for_each_value implementation for the shape may be able to statically
   * optimize shape. The default implementation optimizes the shape at runtime,
   * and the only attempts to convert the shape to a dense_shape. */
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

/** Copy shape traits enable some behaviors to be overriden on a pairwise shape
 * basis for copies. */
template <class ShapeSrc, class ShapeDst = ShapeSrc>
class copy_shape_traits {
 public:
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

/** Iterate over all indices in the shape, calling a function `fn` for each set
 * of indices. The order is defined by `shape_traits<Shape>`. `for_all_indices`
 * calls `fn` with a list of arguments corresponding to each dim.
 * `for_each_index` calls `fn` with a Shape::index_type object describing the
 * indices. */
template <class Shape, class Fn,
    class = internal::enable_if_callable<Fn, typename Shape::index_type>>
void for_each_index(const Shape& s, Fn&& fn) {
  shape_traits<Shape>::for_each_index(s, fn);
}
template <class Shape, class Fn>
void for_all_indices(const Shape& s, Fn&& fn) {
  shape_traits<Shape>::for_each_index(s, [&](const typename Shape::index_type&i) {
    internal::apply(fn, i);
  });
}

template <class T, class Shape>
class array_ref;
template <class T, class Shape, class Alloc>
class array;

/** Make a new array with shape `shape`, allocated using `alloc`. */
template <class T, class Shape>
array_ref<T, Shape> make_array_ref(T* base, const Shape& shape) {
  return {base, shape};
}

namespace internal {

template <class T, class Shape, class... Args>
auto make_array_ref_at(T base, const Shape& shape, Args... args) {
  auto new_shape = shape(args...);
  auto new_mins = mins_of_ranges(
      std::make_tuple(args...), shape.dims(), make_index_sequence<sizeof...(Args)>());
  auto old_min_offset = shape(new_mins);
  return make_array_ref(internal::pointer_add(base, old_min_offset), new_shape);
}

}  // namespace internal

/** A reference to an array is an object with a shape mapping indices to flat
 * offsets, which are used to dereference a pointer. This object has 'reference
 * semantics':
 * - O(1) copy construction, cheap to pass by value.
 * - Cannot be reassigned. */
template <class T, class Shape>
class array_ref {
 public:
  /** Type of elements referenced in this array_ref. */
  using value_type = T;
  using reference = value_type&;
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
  using enable_if_ranges = typename std::enable_if<
      internal::all_of_type<range<>, Args...>::value &&
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
  /** The copy constructor of a ref is a shallow copy. */
  array_ref(const array_ref& other) = default;
  array_ref(array_ref&& other) = default;
  template <class OtherShape, class = enable_if_shape_compatible<OtherShape>>
  array_ref(const array_ref<T, OtherShape>& other) : array_ref(other.base(), other.shape()) {}

  /** Assigning an array_ref is a shallow assignment. */
  array_ref& operator=(const array_ref& other) = default;
  array_ref& operator=(array_ref&& other) = default;
  template <class OtherShape, class = enable_if_shape_compatible<OtherShape>>
  array_ref& operator=(const array_ref<T, OtherShape>& other) {
    base_ = other.base();
    shape_ = other.shape();
    return *this;
  }

  /** Get a reference to the element at the given indices. */
  reference operator() (const index_type& indices) const { return base_[shape_(indices)]; }
  reference operator[] (const index_type& indices) const { return base_[shape_(indices)]; }
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_indices<Args...>>
  reference operator() (Args... indices) const { return base_[shape_(indices...)]; }

  /** Create an array_ref from this array_ref using a series of crops and slices `args`.
   * The resulting array_ref will have the same rank as this array_ref. */
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_ranges<Args...>>
  auto operator() (Args... args) const {
    return internal::make_array_ref_at(base_, shape_, args...);
  }

  /** Call a function with a reference to each value in this array_ref. The
   * order in which `fn` is called is undefined. */
  template <class Fn>
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

  /** Provide some aliases for common interpretations of dimensions. */
  auto& i() { return shape_.i(); }
  const auto& i() const { return shape_.i(); }
  auto& j() { return shape_.j(); }
  const auto& j() const { return shape_.j(); }
  auto& k() { return shape_.k(); }
  const auto& k() const { return shape_.k(); }

  auto& x() { return shape_.x(); }
  const auto& x() const { return shape_.x(); }
  auto& y() { return shape_.y(); }
  const auto& y() const { return shape_.y(); }
  auto& z() { return shape_.z(); }
  const auto& z() const { return shape_.z(); }
  auto& w() { return shape_.w(); }
  const auto& w() const { return shape_.w(); }

  auto& c() { return shape_.c(); }
  const auto& c() const { return shape_.c(); }

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
  bool operator!=(const array_ref& other) const {
    if (shape_.min() != other.shape_.min() || shape_.extent() != other.shape_.extent()) {
      return true;
    }

    // TODO: This currently calls operator!= on all elements of the array_ref,
    // even after we find a non-equal element
    // (https://github.com/dsharlet/array/issues/4).
    bool result = false;
    copy_shape_traits<Shape, Shape>::for_each_value(
        shape_, base_, other.shape_, other.base_, [&](const value_type& a, const value_type& b) {
      if (a != b) {
        result = true;
      }
    });
    return result;
  }
  bool operator==(const array_ref& other) const { return !operator!=(other); }

  const array_ref<T, Shape>& ref() const { return *this; }
  const array_ref<const T, Shape> cref() const { return array_ref<const T, Shape>(base_, shape_); }

  /** Allow conversion from array_ref<T> to array_ref<const T>. */
  operator array_ref<const T, Shape>() const { return cref(); }

  /** Change the shape of the array to `new_shape`, and move the base pointer by
   * `offset`. */
  void set_shape(const Shape& new_shape, index_t offset = 0) {
    assert(new_shape.is_known());
    assert(new_shape.is_subset_of(shape_, -offset));
    shape_ = new_shape;
    base_ = internal::pointer_add(base_, offset);
  }
};

/** array_ref with an arbitrary shape of the compile-time constant `Rank`. */
template <class T, size_t Rank>
using array_ref_of_rank = array_ref<T, shape_of_rank<Rank>>;

/** array_ref with a `dense_dim` innermost dimension, and an arbitrary shape
 * otherwise, of the compile-time constant `Rank`. */
template <class T, size_t Rank>
using dense_array_ref = array_ref<T, dense_shape<Rank>>;

/** A multi-dimensional array container that owns an allocation of memory. This
 * container is designed to mirror the semantics of std::vector where possible.
 */
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
  using copy_shape_traits_type = copy_shape_traits<Shape>;
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
  using enable_if_ranges = typename std::enable_if<
      internal::all_of_type<range<>, Args...>::value &&
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
        other.shape(), other.base(), shape_, base_, [&](const value_type& src, value_type& dst) {
      alloc_traits::construct(alloc_, &dst, src);
    });
  }
  void move_construct(array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape());
    copy_shape_traits_type::for_each_value(
        other.shape(), other.base(), shape_, base_, [&](value_type& src, value_type& dst) {
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
  /** Construct an array with a default constructed Shape. Most shapes by
   * default are empty, but a Shape with non-zero compile-time constants for all
   * extents will be non-empty.
   *
   * When constructing arrays, unknown extents are set to 0, and unknown
   * strides are set to the currently largest known stride. This is done in
   * innermost-to-outermost order. */
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
  array(array&& other, const Alloc& alloc) : array(alloc) {
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

  /** Assign the contents of this array to have `shape` with each element copy
   * constructed from `value`. */
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

  /** Get a reference to the element at the given indices. */
  reference operator() (const index_type& indices) { return base_[shape_(indices)]; }
  reference operator[] (const index_type& indices) { return base_[shape_(indices)]; }
  const_reference operator() (const index_type& indices) const { return base_[shape_(indices)]; }
  const_reference operator[] (const index_type& indices) const { return base_[shape_(indices)]; }
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_indices<Args...>>
  reference operator() (Args... indices) { return base_[shape_(indices...)]; }
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_indices<Args...>>
  const_reference operator() (Args... indices) const { return base_[shape_(indices...)]; }

  /** Create an `array_ref` from this array from a series of crops and slices `args`.
   * The resulting `array_ref` will have the same rank as this array. */
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_ranges<Args...>>
  auto operator() (Args... args) {
    return internal::make_array_ref_at(base_, shape_, args...);
  }
  template <class... Args,
      class = enable_if_same_rank<Args...>,
      class = enable_if_ranges<Args...>>
  auto operator() (Args... args) const {
    return internal::make_array_ref_at(base_, shape_, args...);
  }

  /** Call a function with a reference to each value in this array. The order in
   * which `fn` is called is undefined. */
  template <class Fn>
  void for_each_value(Fn&& fn) {
    shape_traits_type::for_each_value(shape_, base_, fn);
  }
  template <class Fn>
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

  /** Reset the shape of this array to default. If the default Shape is
   * non-empty, the elements of the array will be default constructed. */
  void clear() {
    deallocate();
    shape_ = Shape();
    allocate();
    construct();
  }

  /** Reallocate the array, and move the intersection of the old and new shapes
   * to the new array. */
  void reshape(const Shape& new_shape) {
    if (shape_ == new_shape) {
      return;
    }

    // Allocate an array with the new shape.
    array new_array(new_shape);

    // Move the common elements to the new array.
    Shape intersection = intersect(shape_, new_array.shape());
    copy_shape_traits_type::for_each_value(
        shape_, base_, intersection, new_array.base(), [](T& src, T& dst) {
      dst = std::move(src);
    });

    *this = std::move(new_array);
  }

  /** <b>This function is unsafe and should not be used
   * (https://github.com/dsharlet/array/issues/19).</b> Change the shape
   * of the array to `new_shape`, and move the base pointer by `offset`. */
  void set_shape(const Shape& new_shape, index_t offset = 0) {
    assert(new_shape.is_known());
    assert(new_shape.is_subset_of(shape_, -offset));
    shape_ = new_shape;
    base_ = internal::pointer_add(base_, offset);
  }

  /** Provide some aliases for common interpretations of dimensions. */
  const auto& i() const { return shape_.i(); }
  const auto& j() const { return shape_.j(); }
  const auto& k() const { return shape_.k(); }

  const auto& x() const { return shape_.x(); }
  const auto& y() const { return shape_.y(); }
  const auto& z() const { return shape_.z(); }
  const auto& w() const { return shape_.w(); }

  const auto& c() const { return shape_.c(); }

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
  array_ref<const T, Shape> cref() const { return array_ref<const T, Shape>(base_, shape_); }
  array_ref<const T, Shape> ref() const { return cref(); }
  operator array_ref<T, Shape>() { return ref(); }
  operator array_ref<const T, Shape>() const { return cref(); }
};

/** An array type with an arbitrary shape of rank `Rank`. */
template <class T, size_t Rank, class Alloc = std::allocator<T>>
using array_of_rank = array<T, shape_of_rank<Rank>, Alloc>;

/** array with a `dense_dim` innermost dimension, and an arbitrary shape
 * otherwise, of rank `Rank`. */
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
 * array_ref. The range of the shape of `dst` will be copied, and must be in
 * bounds of `src`. */
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void copy(const array_ref<TSrc, ShapeSrc>& src, const array_ref<TDst, ShapeDst>& dst) {
  if (dst.shape().empty()) {
    return;
  }
  if (!src.shape().is_in_range(dst.shape().min()) ||
      !src.shape().is_in_range(dst.shape().max())) {
    NDARRAY_THROW_OUT_OF_RANGE("dst indices out of range of src");
  }

  copy_shape_traits<ShapeSrc, ShapeDst>::for_each_value(
      src.shape(), src.base(), dst.shape(), dst.base(), [](const TSrc& src_i, TDst& dst_i) {
    dst_i = src_i;
  });
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
 * array_ref. The range of the shape of `dst` will be moved, and must be in
 * bounds of `src`. */
template <class TSrc, class TDst, class ShapeSrc, class ShapeDst,
    class = internal::enable_if_shapes_copy_compatible<ShapeDst, ShapeSrc>>
void move(const array_ref<TSrc, ShapeSrc>& src, const array_ref<TDst, ShapeDst>& dst) {
  if (dst.shape().empty()) {
    return;
  }
  if (!src.shape().is_in_range(dst.shape().min()) ||
      !src.shape().is_in_range(dst.shape().max())) {
    NDARRAY_THROW_OUT_OF_RANGE("dst indices out of range of src");
  }

  copy_shape_traits<ShapeSrc, ShapeDst>::for_each_value(
      src.shape(), src.base(), dst.shape(), dst.base(), [](TSrc& src_i, TDst& dst_i) {
    dst_i = std::move(src_i);
  });
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
auto make_move(const array_ref<T, ShapeSrc>& src, const ShapeDst& shape,
               const Alloc& alloc = Alloc()) {
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

/** Make a copy of the `src` array or array_ref with a compact version of `src`s
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
template <class T, class Shape, class Generator>
void generate(const array_ref<T, Shape>& dst, Generator g) {
  dst.for_each_value([g](T& i) { i = g(); });
}
template <class T, class Shape, class Alloc, class Generator>
void generate(array<T, Shape, Alloc>& dst, Generator g) {
  generate(dst.ref(), g);
}

/** Convert the shape of the array or array_ref `a` to be a new shape
 * `new_shape`. */
template <class NewShape, class T, class OldShape>
array_ref<T, NewShape> convert_shape(const array_ref<T, OldShape>& a) {
  return array_ref<T, NewShape>(a.base(), a.shape());
}
template <class NewShape, class T, class OldShape, class Allocator>
array_ref<T, NewShape> convert_shape(array<T, OldShape, Allocator>& a) {
  return convert_shape<NewShape>(a.ref());
}
template <class NewShape, class T, class OldShape, class Allocator>
array_ref<const T, NewShape> convert_shape(const array<T, OldShape, Allocator>& a) {
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
array_ref<const U, Shape> reinterpret(const array<T, Shape, Alloc>& a) {
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
array_ref<const T, NewShape> reinterpret_shape(
    const array<T, OldShape, Allocator>& a, const NewShape& new_shape, index_t offset = 0) {
  return reinterpret_shape(a.cref(), new_shape, offset);
}

/** Allocator satisfying the std::allocator interface which allocates memory
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
  // TODO: Most of these constructors/assignment operators are hacks,
  // because the C++ STL I'm using seems to not be respecting the
  // propagate typedefs or the 'select_on_...' function above.
  auto_allocator(const auto_allocator&) noexcept : allocated(false) {}
  auto_allocator(auto_allocator&&) noexcept : allocated(false) {}
  auto_allocator& operator=(const auto_allocator&) { return *this; }
  auto_allocator& operator=(auto_allocator&&) { return *this; }

  T* allocate(size_t n) {
    if (allocated) NDARRAY_THROW_BAD_ALLOC();
    if (n > N) NDARRAY_THROW_BAD_ALLOC();
    allocated = true;
    return reinterpret_cast<T*>(&buffer[0]);
  }
  void deallocate(T*, size_t) noexcept {
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

}  // namespace nda

#endif  // NDARRAY_ARRAY_H

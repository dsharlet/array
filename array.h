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

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>

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

typedef std::size_t size_t;
/** When NDARRAY_INT_INDICES is defined, array indices are 'int' values, otherwise
 * they are 'std::ptrdiff_t' */
#ifdef NDARRAY_INT_INDICES
typedef int index_t;
#else
typedef std::ptrdiff_t index_t;
#endif

/** This value indicates a compile-time constant stride is unknown, and to use
 * the corresponding runtime value instead. */
// It would be better to use a more unreasonable value that would never be
// used in practice. Fortunately, this does not affect correctness, only
// performance, and it is hard to imagine a use case for this where
// performance matters.
constexpr index_t UNK = -9;

#define NDARRAY_CHECK_CONSTRAINT(constant, runtime) \
  assert(constant == runtime || constant == UNK);

namespace internal {

// Given a compile-time static value, reconcile a compile-time static value and
// runtime value.
template <index_t Value>
NDARRAY_INLINE index_t reconcile(index_t value) {
  if (Value != UNK) {
    // It would be nice to assert here that Value == value. But, this is used in
    // the innermost loops, so when asserts are on, this ruins performance. It
    // is also a less helpful place to catch errors like this, because the bug
    // it is catching is caused by an issue much earlier than this, so it is
    // better to assert there instead.
    return Value;
  } else {
    return value;
  }
}

}  // namespace internal

/** An iterator over a range of indices, enabling range-based for loops for
 * indices. */
class dim_iterator {
  index_t i_;

 public:
  dim_iterator(index_t i) : i_(i) {}

  NDARRAY_INLINE bool operator==(const dim_iterator& r) const { return i_ == r.i_; }
  NDARRAY_INLINE bool operator!=(const dim_iterator& r) const { return i_ != r.i_; }

  NDARRAY_INLINE index_t operator *() const { return i_; }

  NDARRAY_INLINE dim_iterator operator++(int) { return dim_iterator(i_++); }
  NDARRAY_INLINE dim_iterator& operator++() { ++i_; return *this; }
};

/** Describes one dimension of an array. The template parameters enable
 * providing compile time constants for the 'min', 'extent', and 'stride' of the
 * dim. These parameters define a mapping from the indices of the dimension to
 * offsets: offset(x) = (x - min)*stride. The extent does not affect the mapping
 * directly. Values not in the range [min, min + extent) are considered to be
 * out of bounds. */
// TODO: Consider adding helper class constant<Value> to use for the members of
// dim. (https://github.com/dsharlet/array/issues/1)
template <index_t Min_ = UNK, index_t Extent_ = UNK, index_t Stride_ = UNK>
class dim {
 protected:
  index_t min_;
  index_t extent_;
  index_t stride_;

 public:
  static constexpr index_t Min = Min_;
  static constexpr index_t Extent = Extent_;
  static constexpr index_t Max = Min != UNK && Extent != UNK ? Min + Extent - 1 : UNK;
  static constexpr index_t Stride = Stride_;

  /** Construct a new dim object. If the class template parameters 'Min',
   * 'Extent', or 'Stride' are not 'UNK', these runtime values must match the
   * compile-time values. */
  dim(index_t min, index_t extent, index_t stride = Stride)
    : min_(min), extent_(extent), stride_(stride) {
    NDARRAY_CHECK_CONSTRAINT(Min, min);
    NDARRAY_CHECK_CONSTRAINT(Extent, extent);
    NDARRAY_CHECK_CONSTRAINT(Stride, stride);
  }
  dim(index_t extent = Extent) : dim(0, extent) {}
  dim(const dim&) = default;
  dim(dim&&) = default;
  /** Copy another dim object, possibly with different compile-time template
   * parameters. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride>
  dim(const dim<CopyMin, CopyExtent, CopyStride>& other)
      : dim(other.min(), other.extent(), other.stride()) {
    // We can statically check the compile-time constants, which produces a
    // more convenient compiler error instead of a runtime error.
    static_assert(Min == UNK || CopyMin == UNK || Min == CopyMin,
                  "incompatible mins.");
    static_assert(Extent == UNK || CopyExtent == UNK || Extent == CopyExtent,
                  "incompatible extents.");
    static_assert(Stride == UNK || CopyStride == UNK || Stride == CopyStride,
                  "incompatible strides.");
  }

  dim& operator=(const dim&) = default;
  dim& operator=(dim&&) = default;
  /** Copy assignment of a dim object, possibly with different compile-time
   * template parameters. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride>
  dim& operator=(const dim<CopyMin, CopyExtent, CopyStride>& other) {
    // We can statically check the compile-time constants, which produces a
    // more convenient compiler error instead of a runtime error.
    static_assert(Min == UNK || CopyMin == UNK || Min == CopyMin,
                  "incompatible mins.");
    static_assert(Extent == UNK || CopyExtent == UNK || Extent == CopyExtent,
                  "incompatible extents.");
    static_assert(Stride == UNK || CopyStride == UNK || Stride == CopyStride,
                  "incompatible strides.");

    // Also check the runtime values.
    NDARRAY_CHECK_CONSTRAINT(Min, other.min());
    min_ = other.min();
    NDARRAY_CHECK_CONSTRAINT(Extent, other.extent());
    extent_ = other.extent();
    NDARRAY_CHECK_CONSTRAINT(Stride, other.stride());
    stride_ = other.stride();
    return *this;
  }

  /** Index of the first element in this dim. */
  NDARRAY_INLINE index_t min() const { return internal::reconcile<Min>(min_); }
  void set_min(index_t min) {
    NDARRAY_CHECK_CONSTRAINT(Min, min);
    min_ = min;
  }
  /** Number of elements in this dim. */
  NDARRAY_INLINE index_t extent() const { return internal::reconcile<Extent>(extent_); }
  void set_extent(index_t extent) {
    NDARRAY_CHECK_CONSTRAINT(Extent, extent);
    extent_ = extent;
  }
  /** Distance in flat indices between neighboring elements in this dim. */
  NDARRAY_INLINE index_t stride() const { return internal::reconcile<Stride>(stride_); }
  void set_stride(index_t stride) {
    NDARRAY_CHECK_CONSTRAINT(Stride, stride);
    stride_ = stride;
  }
  /** Index of the last element in this dim. */
  NDARRAY_INLINE index_t max() const { return min() + extent() - 1; }

  /** Offset of the index 'at' in this dim in the flat array. */
  NDARRAY_INLINE index_t flat_offset(index_t at) const { return (at - min()) * stride(); }

  /** Returns true if 'at' is within the range [min(), max()]. */
  NDARRAY_INLINE bool is_in_range(index_t at) const { return min() <= at && at <= max(); }

  /** Make an iterator referring to the first element in this dim. */
  dim_iterator begin() const { return dim_iterator(min()); }
  /** Make an iterator referring to one past the last element in this dim. */
  dim_iterator end() const { return dim_iterator(max() + 1); }

  /** Two dim objects are considered equal if their mins, extents, and strides
   * are equal. */
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride>
  bool operator==(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
    return min() == other.min() && extent() == other.extent() && stride() == other.stride();
  }
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride>
  bool operator!=(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
    return !operator==(other);
  }
};

template <index_t Min, index_t Extent, index_t Stride>
dim_iterator begin(const dim<Min, Extent, Stride>& d) {
  return d.begin();
}
template <index_t Min, index_t Extent, index_t Stride>
dim_iterator end(const dim<Min, Extent, Stride>& d) {
  return d.end();
}

/** A specialization of 'dim' where the compile-time stride parameter is known
 * to be one. */
template <index_t Min = UNK, index_t Extent = UNK>
using dense_dim = dim<Min, Extent, 1>;

/** A specialization of 'dim' where only the stride parameter is specified at
 * compile time. */
template <index_t Stride>
using strided_dim = dim<UNK, UNK, Stride>;

/** A specialization of 'dim' where the compile-time stride parameter is known
 * to be zero. */
template <index_t Min = UNK, index_t Extent = UNK>
using broadcast_dim = dim<Min, Extent, 0>;

/** Clamp an index to the range [min, max]. */
inline index_t clamp(index_t x, index_t min, index_t max) {
  return std::min(std::max(x, min), max);
}

/** Clamp an index to the range described by a dim. */
template <typename Dim>
index_t clamp(index_t x, const Dim& d) {
  return clamp(x, d.min(), d.max());
}

namespace internal {

// Some variadic reduction helpers.
NDARRAY_INLINE index_t sum() { return 0; }
NDARRAY_INLINE index_t product() { return 1; }
NDARRAY_INLINE index_t variadic_min() { return std::numeric_limits<index_t>::max(); }
NDARRAY_INLINE index_t variadic_max() { return std::numeric_limits<index_t>::min(); }

template <typename... Rest>
NDARRAY_INLINE index_t sum(index_t first, Rest... rest) {
  return first + sum(rest...);
}

template <typename... Rest>
NDARRAY_INLINE index_t product(index_t first, Rest... rest) {
  return first * product(rest...);
}

template <typename... Rest>
NDARRAY_INLINE index_t variadic_min(index_t first, Rest... rest) {
  return std::min(first, variadic_min(rest...));
}

template <typename... Rest>
NDARRAY_INLINE index_t variadic_max(index_t first, Rest... rest) {
  return std::max(first, variadic_max(rest...));
}

// Computes the product of the extents of the dims.
template <typename... Ts, size_t... Is>
index_t product(const std::tuple<Ts...>& t, std::index_sequence<Is...>) {
  return product(std::get<Is>(t)...);
}

// Returns true if all of bools are true.
template <typename... Bools>
bool all(Bools... bools) {
  return sum((bools ? 0 : 1)...) == 0;
}

// Computes the sum of the offsets of a list of dims and indices.
template <typename Dims, typename Indices, size_t... Is>
index_t flat_offset_tuple_impl(const Dims& dims, const Indices& indices, std::index_sequence<Is...>) {
  return sum(std::get<Is>(dims).flat_offset(std::get<Is>(indices))...);
}

template <typename Dims, typename Indices>
index_t flat_offset_tuple(const Dims& dims, const Indices& indices) {
  constexpr size_t dims_rank = std::tuple_size<Dims>::value;
  constexpr size_t indices_rank = std::tuple_size<Indices>::value;
  static_assert(dims_rank == indices_rank, "dims and indices must have the same rank.");
  return flat_offset_tuple_impl(dims, indices, std::make_index_sequence<dims_rank>());
}

template <size_t D, typename Dims>
index_t flat_offset_pack(const Dims& dims) {
  constexpr size_t dims_rank = std::tuple_size<Dims>::value;
  static_assert(dims_rank == D, "dims and indices must have the same rank.");
  return 0;
}

template <size_t D, typename Dims, typename... Indices>
index_t flat_offset_pack(const Dims& dims, index_t i0, Indices... indices) {
  return std::get<D>(dims).flat_offset(i0) + flat_offset_pack<D + 1>(dims, indices...);
}

// Computes one more than the sum of the offsets of the last index in every dim.
template <typename Dims, size_t... Is>
index_t flat_min(const Dims& dims, std::index_sequence<Is...>) {
  return sum((std::get<Is>(dims).extent() - 1) *
              std::min(static_cast<index_t>(0), std::get<Is>(dims).stride())...);
}

template <typename Dims, size_t... Is>
index_t flat_max(const Dims& dims, std::index_sequence<Is...>) {
  return sum((std::get<Is>(dims).extent() - 1) *
              std::max(static_cast<index_t>(0), std::get<Is>(dims).stride())...);
}

// Checks if all indices are in range of each corresponding dim.
template <typename Dims, typename Indices, size_t... Is>
index_t is_in_range_impl(const Dims& dims, const Indices& indices, std::index_sequence<Is...>) {
  return all(std::get<Is>(dims).is_in_range(std::get<Is>(indices))...);
}

template <typename Dims, typename Indices>
bool is_in_range(const Dims& dims, const Indices& indices) {
  constexpr size_t dims_rank = std::tuple_size<Dims>::value;
  constexpr size_t indices_rank = std::tuple_size<Indices>::value;
  static_assert(dims_rank == indices_rank, "dims and indices must have the same rank.");
  return is_in_range_impl(dims, indices, std::make_index_sequence<dims_rank>());
}

template <typename... Dims, size_t... Is>
auto mins(const std::tuple<Dims...>& dims, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).min()...);
}

template <typename... Dims, size_t... Is>
auto extents(const std::tuple<Dims...>& dims, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).extent()...);
}

template <typename... Dims, size_t... Is>
auto strides(const std::tuple<Dims...>& dims, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).stride()...);
}

template <typename... Dims, size_t... Is>
auto maxs(const std::tuple<Dims...>& dims, std::index_sequence<Is...>) {
  return std::make_tuple(std::get<Is>(dims).max()...);
}

template <typename Dim>
bool is_dim_ok(index_t stride, index_t extent, const Dim& dim) {
  if (dim.stride() == UNK) {
    // If the dimension has an unknown stride, it's OK, we're resolving the
    // current dim first.
    return true;
  }
  if (dim.extent() * std::abs(dim.stride()) <= stride) {
    // The dim is completely inside the proposed stride.
    return true;
  }
  index_t flat_extent = extent * stride;
  if (std::abs(dim.stride()) >= flat_extent) {
    // The dim is completely outside the proposed stride.
    return true;
  }
  return false;
}

template <typename AllDims, size_t... Is>
bool is_dim_ok(index_t stride, index_t extent, const AllDims& all_dims, std::index_sequence<Is...>) {
  return all(is_dim_ok(stride, extent, std::get<Is>(all_dims))...);
}

template <typename AllDims>
index_t filter_stride(index_t stride, index_t extent, const AllDims& all_dims) {
  constexpr size_t rank = std::tuple_size<AllDims>::value;
  if (is_dim_ok(stride, extent, all_dims, std::make_index_sequence<rank>())) {
    return stride;
  } else {
    return std::numeric_limits<index_t>::max();
  }
}

template <size_t D, typename AllDims>
index_t candidate_stride(index_t extent, const AllDims& all_dims) {
  index_t stride_d = std::get<D>(all_dims).stride();
  if (stride_d == UNK) {
    return std::numeric_limits<index_t>::max();
  }
  index_t stride = std::max(static_cast<index_t>(1), std::abs(stride_d) * std::get<D>(all_dims).extent());
  return filter_stride(stride, extent, all_dims);
}

template <typename AllDims, size_t... Is>
index_t find_stride(index_t extent, const AllDims& all_dims, std::index_sequence<Is...>) {
  return variadic_min(filter_stride(1, extent, all_dims),
                      candidate_stride<Is>(extent, all_dims)...);
}

inline void resolve_unknown_extents() {}

template <typename Dim0, typename... Dims>
void resolve_unknown_extents(Dim0& dim0, Dims&... dims) {
  if (dim0.extent() == UNK) {
    dim0.set_extent(0);
  }
  resolve_unknown_extents(dims...);
}

template <typename Dims, size_t... Is>
void resolve_unknown_extents(Dims& dims, std::index_sequence<Is...>) {
  resolve_unknown_extents(std::get<Is>(dims)...);
}


template <typename AllDims>
void resolve_unknown_strides(AllDims& all_dims) {}

template <typename AllDims, typename Dim0, typename... Dims>
void resolve_unknown_strides(AllDims& all_dims, Dim0& dim0, Dims&... dims) {
  if (dim0.stride() == UNK) {
    constexpr size_t rank = std::tuple_size<AllDims>::value;
    dim0.set_stride(find_stride(dim0.extent(), all_dims, std::make_index_sequence<rank>()));
  }
  resolve_unknown_strides(all_dims, dims...);
}

template <typename Dims, size_t... Is>
void resolve_unknown_strides(Dims& dims, std::index_sequence<Is...>) {
  resolve_unknown_strides(dims, std::get<Is>(dims)...);
}

template <typename Dims>
void resolve_unknowns(Dims& dims) {
  constexpr size_t rank = std::tuple_size<Dims>::value;
  resolve_unknown_extents(dims, std::make_index_sequence<rank>());
  resolve_unknown_strides(dims, std::make_index_sequence<rank>());
}

// A helper to transform an array to a tuple.
template <typename T, typename... Ts, size_t... Is>
std::array<T, sizeof...(Is)> tuple_to_array(const std::tuple<Ts...>& t, std::index_sequence<Is...>) {
  return {{std::get<Is>(t)...}};
}

template <typename T, typename... Ts>
std::array<T, sizeof...(Ts)> tuple_to_array(const std::tuple<Ts...>& t) {
  return tuple_to_array<T>(t, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T, size_t N, size_t... Is>
auto array_to_tuple(const std::array<T, N>& a, std::index_sequence<Is...>) {
  return std::make_tuple(a[Is]...);
}

template <typename T, size_t N>
auto array_to_tuple(const std::array<T, N>& a) {
  return array_to_tuple(a, std::make_index_sequence<N>());
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

template <typename... New, typename... Old, size_t... Is>
std::tuple<New...> convert_tuple(const std::tuple<Old...>& in, std::index_sequence<Is...>) {
  return std::tuple<New...>(std::get<Is>(in)...);
}

template <typename... New, typename... Old>
std::tuple<New...> convert_tuple(const std::tuple<Old...>& in) {
  static_assert(sizeof...(New) == sizeof...(Old), "tuple conversion of differently sized tuples");
  return convert_tuple<New...>(in, std::make_index_sequence<sizeof...(Old)>());
}

// https://github.com/halide/Halide/blob/fc8cfb078bed19389f72883a65d56d979d18aebe/src/runtime/HalideBuffer.h#L43-L63
// A helper to check if a parameter pack is entirely implicitly int-convertible
// to use with std::enable_if
template<typename... Args>
struct all_integral : std::false_type {};

template<>
struct all_integral<> : std::true_type {};

template<typename T, typename... Args>
struct all_integral<T, Args...> {
  static constexpr bool value =
      std::is_convertible<T, index_t>::value && all_integral<Args...>::value;
};

}  // namespace internal

/** A list of 'dim' objects describing a multi-dimensional space of indices.
 * The 'rank' of a shape refers to the number of dimensions in the shape.
 * Shapes map multiple dim objects to offsets by adding each mapping dim to
 * offset together to produce a 'flat offset'. The first dimension is known as
 * the 'innermost' dimension, and dimensions then increase until the
 * 'outermost' dimension. */
template <typename... Dims>
class shape {
  std::tuple<Dims...> dims_;

 public:
  /** When constructing shapes, unknown extents are set to 0, and unknown
   * strides are set to the currently largest known stride. This is done in
   * innermost-to-outermost order. */
  shape() { internal::resolve_unknowns(dims_); }
  shape(std::tuple<Dims...> dims) : dims_(std::move(dims)) { internal::resolve_unknowns(dims_); }
  shape(Dims... dims) : dims_(std::move(dims)...) { internal::resolve_unknowns(dims_); }
  shape(const shape&) = default;
  shape(shape&&) = default;
  /** Construct this shape from a different type of shape. 'conversion' must
   * be convertible to this shape. */
  template <typename... OtherDims>
  shape(const shape<OtherDims...>& conversion)
    : dims_(internal::convert_tuple<Dims...>(conversion.dims())) {}

  shape& operator=(const shape&) = default;
  shape& operator=(shape&&) = default;

  /** Assign this shape from a different type of shape. 'conversion' must be
   * convertible to this shape. */
  template <typename... OtherDims>
  shape& operator=(const shape<OtherDims...>& conversion) {
    dims_ = internal::convert_tuple<Dims...>(conversion.dims());
    return *this;
  }

  /** Number of dims in this shape. */
  static constexpr size_t rank() { return std::tuple_size<std::tuple<Dims...>>::value; }

  /** A shape is scalar if it is rank 0. */
  static constexpr bool is_scalar() { return rank() == 0; }

  /** The type of an index for this shape. */
  typedef typename internal::tuple_of_n<index_t, rank()>::type index_type;

  /** Returns true if the index 'indices' are in range of this shape. */
  bool is_in_range(const index_type& indices) const {
    return internal::is_in_range(dims_, indices);
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  bool is_in_range(Indices... indices) const {
    return is_in_range(std::make_tuple(indices...));
  }

  /** Compute the flat offset of the index 'indices'. */
  index_t operator() (const index_type& indices) const { return internal::flat_offset_tuple(dims_, indices); }
  index_t operator[] (const index_type& indices) const { return internal::flat_offset_tuple(dims_, indices); }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  index_t operator() (Indices... indices) const {
    return internal::flat_offset_pack<0>(dims_, indices...);
  }

  /** Get a specific dim of this shape. */
  template <size_t D>
  auto& dim() { return std::get<D>(dims_); }
  template <size_t D>
  const auto& dim() const { return std::get<D>(dims_); }
  /** Get a specific dim of this shape with a runtime dimension d. This will
   * lose knowledge of any compile-time constant dimension attributes. */
  nda::dim<> dim(size_t d) const {
    return internal::tuple_to_array<nda::dim<>>(dims_)[d];
  }

  /** Get a tuple of all of the dims of this shape. */
  std::tuple<Dims...>& dims() { return dims_; }
  const std::tuple<Dims...>& dims() const { return dims_; }

  /** Get an index pointing to the min or max index in each dimension of this
   * shape. */
  index_type min() const {
    return internal::mins(dims(), std::make_index_sequence<rank()>());
  }
  index_type max() const {
    return internal::maxs(dims(), std::make_index_sequence<rank()>());
  }
  index_type extent() const {
    return internal::extents(dims(), std::make_index_sequence<rank()>());
  }
  index_type stride() const {
    return internal::strides(dims(), std::make_index_sequence<rank()>());
  }

  /** Compute the flat extent of this shape. This is the extent of the valid
   * range of values returned by at or operator(). */
  index_t flat_min() const {
    return internal::flat_min(dims_, std::make_index_sequence<rank()>());
  }
  index_t flat_max() const {
    return internal::flat_max(dims_, std::make_index_sequence<rank()>());
  }
  size_t flat_extent() const {
    index_t e = flat_max() - flat_min() + 1;
    return e < 0 ? 0 : static_cast<size_t>(e);
  }

  /** Compute the total number of items in the shape. */
  size_t size() const {
    index_t s = internal::product(extent(), std::make_index_sequence<rank()>());
    return s < 0 ? 0 : static_cast<size_t>(s);
  }

  /** A shape is empty if its size is 0. */
  bool empty() const { return size() == 0; }

  /** Returns true if this shape is 'compact' in memory. A shape is compact if
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
  bool is_subset_of(const shape& other, index_t offset) const {
    // TODO: https://github.com/dsharlet/array/issues/2
    return
        flat_min() >= other.flat_min() + offset &&
        flat_max() <= other.flat_max() + offset;
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

  /** A shape is equal to another shape if both of the dim objects of
   * each dimension from both shapes are equal. */
  template <typename... OtherDims>
  bool operator==(const shape<OtherDims...>& other) const { return dims_ == other.dims(); }
  template <typename... OtherDims>
  bool operator!=(const shape<OtherDims...>& other) const { return dims_ != other.dims(); }
};

// TODO: Try to avoid needing this specialization
// (https://github.com/dsharlet/array/issues/3).
template <>
class shape<> {
 public:
  shape() {}

  static constexpr size_t rank() { return 0; }
  static constexpr bool is_scalar() { return true; }

  typedef std::tuple<> index_type;

  bool is_in_range(const std::tuple<>&) const { return true; }
  bool is_in_range() const { return true; }

  index_t operator() (const std::tuple<>&) const { return 0; }
  index_t operator[] (const std::tuple<>&) const { return 0; }
  index_t operator() () const { return 0; }

  nda::dim<> dim(size_t) const { return nda::dim<>(); }
  std::tuple<> dims() const { return std::tuple<>(); }

  index_type min() const { return std::tuple<>(); }
  index_type max() const { return std::tuple<>(); }
  index_type extent() const { return std::tuple<>(); }

  index_t flat_min() const { return 0; }
  index_t flat_max() const { return 0; }
  size_t flat_extent() const { return 1; }
  size_t size() const { return 1; }
  bool empty() const { return false; }

  bool is_subset_of(const shape<>&) const { return true; }
  bool is_one_to_one() const { return true; }
  bool is_compact() const { return true; }

  bool operator==(const shape<>&) const { return true; }
  bool operator!=(const shape<>&) const { return false; }
};

/** Helper function to make a tuple from a variadic list of dims. */
template <typename... Dims>
auto make_shape(Dims... dims) {
  return shape<Dims...>(std::forward<Dims>(dims)...);
}

/** Helper function to make a dense shape from a variadic list of extents. */
template <typename... Extents>
auto make_dense_shape(index_t dim0_extent, Extents... extents) {
  return make_shape(dense_dim<>(dim0_extent), dim<>(extents)...);
}

namespace internal {

template<size_t D, typename Dims, typename Fn, typename... Indices,
  std::enable_if_t<(D == 0), int> = 0>
void for_each_index_in_order(const Dims& dims, Fn&& fn, const std::tuple<Indices...>& indices) {
  for (index_t i : std::get<D>(dims)) {
    fn(std::tuple_cat(std::make_tuple(i), indices));
  }
}

template<size_t D, typename Dims, typename Fn, typename... Indices,
  std::enable_if_t<(D > 0), int> = 0>
void for_each_index_in_order(const Dims& dims, Fn&& fn, const std::tuple<Indices...>& indices) {
  for (index_t i : std::get<D>(dims)) {
    for_each_index_in_order<D - 1>(dims, fn, std::tuple_cat(std::make_tuple(i), indices));
  }
}

template <size_t D>
NDARRAY_INLINE void advance() {}

template <size_t D, typename Ptr, typename... Ptrs>
NDARRAY_INLINE void advance(Ptr& ptr, Ptrs&... ptrs) {
  std::get<0>(ptr) += std::get<D>(std::get<1>(ptr));
  advance<D>(ptrs...);
}

template <size_t D, typename ExtentType, typename Fn, typename... Ptrs,
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

template <size_t D, typename ExtentType, typename Fn, typename... Ptrs,
  std::enable_if_t<(D > 0), int> = 0>
void for_each_value_in_order(const ExtentType& extent, Fn&& fn, Ptrs... ptrs) {
  index_t extent_d = std::get<D>(extent);
  for (index_t i = 0; i < extent_d; i++) {
    for_each_value_in_order<D - 1>(extent, fn, ptrs...);
    advance<D>(ptrs...);
  }
}

template <typename... Dims>
shape<Dims...> make_shape_from_tuple(const std::tuple<Dims...>& dims) {
  return shape<Dims...>(dims);
}

template <size_t Rank, size_t... Is>
auto make_default_dense_shape() {
  return make_shape_from_tuple(std::tuple_cat(std::make_tuple(dense_dim<>()),
                                              typename tuple_of_n<dim<>, Rank - 1>::type()));
}

template <typename Shape, size_t... Is>
auto make_dense_shape(const Shape& dims, std::index_sequence<Is...>) {
  return make_shape(dense_dim<>(std::get<0>(dims).min(), std::get<0>(dims).extent()),
                    dim<>(std::get<Is + 1>(dims).min(), std::get<Is + 1>(dims).extent())...);
}

template <typename... Dims, size_t... Is>
shape<Dims...> make_compact_shape(const shape<Dims...>& s, std::index_sequence<Is...>) {
  return {{s.template dim<Is>().min(), s.template dim<Is>().extent()}...};
}

template <index_t Min, index_t Extent, index_t Stride, typename DimSrc>
bool is_dim_compatible(const dim<Min, Extent, Stride>&, const DimSrc& src) {
  return
    (Min == UNK || src.min() == Min) &&
    (Extent == UNK || src.extent() == Extent) &&
    (Stride == UNK || src.stride() == Stride);
}

template <typename... DimsDest, typename ShapeSrc, size_t... Is>
bool is_shape_compatible(const shape<DimsDest...>&, const ShapeSrc& src, std::index_sequence<Is...>) {
  return all(is_dim_compatible(DimsDest(), src.template dim<Is>())...);
}

template <typename DimA, typename DimB>
auto intersect_dims(const DimA& a, const DimB& b) {
  constexpr index_t Min = DimA::Min == UNK || DimB::Min == UNK ? UNK : (DimA::Min > DimB::Min ? DimA::Min : DimB::Min);
  constexpr index_t Max = DimA::Max == UNK || DimB::Max == UNK ? UNK : (DimA::Max < DimB::Max ? DimA::Max : DimB::Max);
  constexpr index_t Extent = Min == UNK || Max == UNK ? UNK : Max - Min + 1;
  constexpr index_t Stride = DimA::Stride == DimB::Stride ? DimA::Stride : UNK;
  index_t min = std::max(a.min(), b.min());
  index_t max = std::min(a.max(), b.max());
  index_t extent = max - min + 1;
  return dim<Min, Extent, Stride>(min, extent);
}

template <typename... DimsA, typename... DimsB, size_t... Is>
auto intersect(const std::tuple<DimsA...>& a, const std::tuple<DimsB...>& b, std::index_sequence<Is...>) {
  return make_shape(intersect_dims(std::get<Is>(a), std::get<Is>(b))...);
}

// Call 'fn' with the elements of tuple 'args' unwrapped from the tuple.
template <typename Fn, typename IndexType, size_t... Is>
NDARRAY_INLINE auto tuple_arg_to_parameter_pack(Fn&& fn, const IndexType& args, std::index_sequence<Is...>) {
  fn(std::get<Is>(args)...);
}

}  // namespace internal

/** Make a shape with an equivalent domain of indices, with dense strides. */
template <typename... Dims>
auto make_dense(const shape<Dims...>& shape) {
  constexpr int rank = sizeof...(Dims);
  return internal::make_dense_shape(shape.dims(), std::make_index_sequence<rank - 1>());
}

/** Make a shape with an equivalent domain of indices, but with compact strides.
 * Only required strides are respected. */
template <typename Shape>
Shape make_compact(const Shape& s) {
  return internal::make_compact_shape(s, std::make_index_sequence<Shape::rank()>());
}

/** An arbitrary shape (no compile-time constant parameters) with the specified
 * Rank. */
template <size_t Rank>
using shape_of_rank =
  decltype(internal::make_shape_from_tuple(typename internal::tuple_of_n<dim<>, Rank>::type()));

/** A shape where the innermost dimension is a 'dense_dim' object, and all other
 * dimensions are arbitrary. */
template <size_t Rank>
using dense_shape = decltype(internal::make_default_dense_shape<Rank>());

/** Test if a shape 'src' can be assigned to a shape of type 'ShapeDest' without
 * error. */
template <typename ShapeDest, typename ShapeSrc>
bool is_convertible(const ShapeSrc& src) {
  static_assert(ShapeSrc::rank() == ShapeDest::rank(), "shapes must have the same rank.");
  return internal::is_shape_compatible(ShapeDest(), src, std::make_index_sequence<ShapeSrc::rank()>());
}

/** Compute the intersection of two shapes 'a' and 'b'. The intersection is the
 * shape containing the indices in bounds of both 'a' and 'b'. */
template <typename ShapeA, typename ShapeB>
auto intersect(const ShapeA& a, const ShapeB& b) {
  constexpr size_t rank = ShapeA::rank() < ShapeB::rank() ? ShapeA::rank() : ShapeB::rank();
  return internal::intersect(a.dims(), b.dims(), std::make_index_sequence<rank>());
}

/** Iterate over all indices in the shape, calling a function 'fn' for each set
 * of indices. The indices are in the same order as the dims in the shape. The
 * first dim is the 'inner' loop of the iteration, and the last dim is the
 * 'outer' loop.
 *
 * These functions are typically used to implement shape_traits and
 * copy_shape_traits objects. Use for_each_index, array_ref<>::for_each_value,
 * or array<>::for_each_value instead. */
template<typename Shape, typename Fn>
void for_each_index_in_order(const Shape& shape, Fn &&fn) {
  internal::for_each_index_in_order<Shape::rank() - 1>(shape.dims(), fn, std::tuple<>());
}
template<typename Shape, typename Ptr, typename Fn>
void for_each_value_in_order(const Shape& shape, Ptr base, Fn &&fn) {
  typedef typename Shape::index_type index_type;
  // TODO: This is losing compile-time constant extents and strides info
  // (https://github.com/dsharlet/array/issues/1).
  std::tuple<Ptr, index_type> base_and_stride(base, shape.stride());
  internal::for_each_value_in_order<Shape::rank() - 1>(shape.extent(), fn, base_and_stride);
}

/** Similar to for_each_value_in_order, but iterates over two arrays
 * simultaneously. 'shape' defines the loop nest, while 'shape_a' and 'shape_b'
 * define the memory layout of 'base_a' and 'base_b'. */
template<typename Shape, typename ShapeA, typename PtrA, typename ShapeB, typename PtrB, typename Fn>
void for_each_value_in_order(const Shape& shape,
                             const ShapeA& shape_a, PtrA base_a,
                             const ShapeB& shape_b, PtrB base_b,
                             Fn &&fn) {
  base_a += shape_a(shape.min());
  base_b += shape_b(shape.min());
  // TODO: This is losing compile-time constant extents and strides info
  // (https://github.com/dsharlet/array/issues/1).
  typedef typename Shape::index_type index_type;
  std::tuple<PtrA, index_type> a(base_a, shape_a.stride());
  std::tuple<PtrB, index_type> b(base_b, shape_b.stride());
  internal::for_each_value_in_order<Shape::rank() - 1>(shape.extent(), fn, a, b);
}

namespace internal {

inline bool can_fuse(const nda::dim<>& inner, const nda::dim<>& outer) {
  return inner.stride() * inner.extent() == outer.stride();
}

inline nda::dim<> fuse(nda::dim<> inner, const nda::dim<>& outer) {
  inner.set_min(inner.min() + outer.min() * inner.extent());
  inner.set_extent(inner.extent() * outer.extent());
  return inner;
}

// Sort the dims such that strides are increasing from dim 0, and contiguous
// dimensions are fused.
template <typename Shape>
shape_of_rank<Shape::rank()> dynamic_optimize_shape(const Shape& shape) {
  auto dims = internal::tuple_to_array<dim<>>(shape.dims());

  // Sort the dims by stride.
  std::sort(dims.begin(), dims.end(), [](const dim<>& l, const dim<>& r) {
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

// Optimize a src and dest shape. The dest shape is made dense, and contiguous
// dimensions are fused.
template <typename ShapeSrc, typename ShapeDest>
auto dynamic_optimize_copy_shapes(const ShapeSrc& src, const ShapeDest& dest) {
  constexpr size_t rank = ShapeSrc::rank();
  static_assert(rank == ShapeDest::rank(), "copy shapes must have same rank.");
  auto src_dims = internal::tuple_to_array<dim<>>(src.dims());
  auto dest_dims = internal::tuple_to_array<dim<>>(dest.dims());

  struct copy_dims {
    dim<> src;
    dim<> dest;
  };
  std::array<copy_dims, rank> dims;
  for (size_t i = 0; i < rank; i++) {
    dims[i] = {src_dims[i], dest_dims[i]};
  }

  // Sort the dims by the dest stride.
  std::sort(dims.begin(), dims.end(), [](const copy_dims& l, const copy_dims& r) {
    return l.dest.stride() < r.dest.stride();
  });

  // Find dimensions that are contiguous and fuse them.
  size_t new_rank = dims.size();
  for (size_t i = 0; i + 1 < new_rank;) {
    if (dims[i].src.extent() == dims[i].dest.extent() &&
        can_fuse(dims[i].src, dims[i + 1].src) &&
        can_fuse(dims[i].dest, dims[i + 1].dest)) {
      dims[i].src = fuse(dims[i].src, dims[i + 1].src);
      dims[i].dest = fuse(dims[i].dest, dims[i + 1].dest);
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
      dim<>(0, 1, dims[i - 1].dest.stride() * dims[i - 1].dest.extent()),
    };
  }

  for (size_t i = 0; i < dims.size(); i++) {
    src_dims[i] = dims[i].src;
    dest_dims[i] = dims[i].dest;
  }

  return std::make_pair(
    shape_of_rank<rank>(array_to_tuple(src_dims)),
    shape_of_rank<rank>(array_to_tuple(dest_dims)));
}

template <typename T>
T* pointer_add(T* x, index_t offset) {
  return x != nullptr ? x + offset : x;
}

}  // namespace internal

/** Shape traits enable some behaviors to be overriden per shape type. */
template <typename Shape>
class shape_traits {
 public:
  typedef Shape shape_type;

  /** The for_each_index implementation for the shape may choose to iterate in a
   * different order than the default (in-order). */
  template <typename Fn>
  static void for_each_index(const Shape& shape, Fn&& fn) {
    for_each_index_in_order(shape, fn);
  }

  /** The for_each_value implementation for the shape may be able to statically
   * optimize shape. The default implementation optimizes the shape at runtime,
   * and the only attempts to convert the shape to a dense_shape. */
  template <typename Ptr, typename Fn>
  static void for_each_value(const Shape& shape, Ptr base, Fn&& fn) {
    auto opt_shape = internal::dynamic_optimize_shape(shape);
    for_each_value_in_order(opt_shape, base, fn);
  }
};

template <>
class shape_traits<shape<>> {
 public:
  typedef shape<> shape_type;

  template <typename Fn>
  static void for_each_index(const shape<>&, Fn&& fn) {
    fn(std::tuple<>());
  }

  template <typename Ptr, typename Fn>
  static void for_each_value(const shape<>&, Ptr base, Fn&& fn) {
    fn(*base);
  }
};

/** Copy shape traits enable some behaviors to be overriden on a pairwise shape
 * basis for copies. */
template <typename ShapeSrc, typename ShapeDest = ShapeSrc>
class copy_shape_traits {
 public:
  template <typename Fn, typename TSrc, typename TDest>
  static void for_each_value(const ShapeSrc& shape_src, TSrc src,
                             const ShapeDest& shape_dest, TDest dest,
                             Fn&& fn) {
    // For this function, we don't care about the order in which the callback is
    // called. Optimize the shapes for memory access order.
    auto opt_shape = internal::dynamic_optimize_copy_shapes(shape_src, shape_dest);
    const auto& opt_shape_src = opt_shape.first;
    const auto& opt_shape_dest = opt_shape.second;

    for_each_value_in_order(opt_shape_dest, opt_shape_src, src, opt_shape_dest, dest, fn);
  }
};

/** Iterate over all indices in the shape, calling a function 'fn' for each set
 * of indices. The order is defined by 'shape_traits<Shape>'. 'for_all_indices'
 * calls 'fn' with a list of arguments corresponding to each dim.
 * 'for_each_index' calls 'fn' with a Shape::index_type object describing the
 * indices. */
template <typename Shape, typename Fn>
void for_each_index(const Shape& s, Fn&& fn) {
  shape_traits<Shape>::for_each_index(s, fn);
}
template <typename Shape, typename Fn>
void for_all_indices(const Shape& s, Fn&& fn) {
  shape_traits<Shape>::for_each_index(s, [&](const typename Shape::index_type&i) {
    internal::tuple_arg_to_parameter_pack(fn, i, std::make_index_sequence<Shape::rank()>());
  });
}

/** Create a new shape using a list of DimIndices to use as the dimensions of
 * the shape. */
template <size_t... DimIndices, typename Shape>
auto select_dims(const Shape& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}
template <size_t... DimIndices, typename Shape>
auto reorder(const Shape& shape) {
  return select_dims<DimIndices...>(shape);
}

/** A reference to an array is an object with a shape mapping indices to flat
 * offsets, which are used to dereference a pointer. This object has 'reference
 * semantics':
 * - O(1) copy construction, cheap to pass by value.
 * - Cannot be reassigned. */
template <typename T, typename Shape>
class array_ref {
 public:
  /** Type of elements referenced in this array_ref. */
  typedef T value_type;
  typedef value_type& reference;
  typedef value_type* pointer;
  /** Type of the shape of this array_ref. */
  typedef Shape shape_type;
  /** Type of the indices used to access this array_ref. */
  typedef typename Shape::index_type index_type;
  typedef size_t size_type;

 private:
  pointer base_;
  Shape shape_;

 public:
  /** Make an array_ref to the given 'base' pointer, interpreting it as having
   * the shape 'shape'. */
  array_ref(pointer base = nullptr, Shape shape = Shape())
      : base_(base), shape_(std::move(shape)) {}
  /** The copy constructor of a ref is a shallow copy. */
  array_ref(const array_ref& other) = default;
  array_ref(array_ref&& other) = default;

  /** Assigning an array_ref is a shallow assignment. */
  array_ref& operator=(const array_ref& other) = default;
  array_ref& operator=(array_ref&& other) = default;

  /** Get a reference to the element at the given indices. */
  reference operator() (const index_type& indices) const { return base_[shape_(indices)]; }
  reference operator[] (const index_type& indices) const { return base_[shape_(indices)]; }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  reference operator() (Indices... indices) const { return base_[shape_(indices...)]; }

  /** Call a function with a reference to each value in this array_ref. The
   * order in which 'fn' is called is undefined. */
  template <typename Fn>
  void for_each_value(Fn&& fn) const {
    shape_traits<Shape>::for_each_value(shape_, base_, fn);
  }

  /** Pointer to the element at the min index of the shape. */
  pointer base() const { return base_; }

  /** Pointer to the element at the beginning of the flat array. */
  pointer data() const { return internal::pointer_add(base_, shape_.flat_min()); }

  /** Shape of this array_ref. */
  const Shape& shape() const { return shape_; }

  static constexpr size_t rank() { return Shape::rank(); }
  static constexpr bool is_scalar() { return Shape::is_scalar(); }
  template <size_t D>
  auto& dim() { return shape_.template dim<D>(); }
  template <size_t D>
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

  /** Compare the contents of this array_ref to 'other'. For two array_refs to
   * be considered equal, they must have the same shape, and all elements
   * addressable by the shape must also be equal. */
  bool operator!=(const array_ref& other) const {
    if (shape_.min() != other.shape_.min() ||
        shape_.extent() != other.shape_.extent()) {
      return true;
    }

    // TODO: This currently calls operator!= on all elements of the array_ref,
    // even after we find a non-equal element
    // (https://github.com/dsharlet/array/issues/4).
    bool result = false;
    for_each_index(shape_, [&](const index_type& i) {
      if ((*this)(i) != other(i)) {
        result = true;
      }
    });
    return result;
  }
  bool operator==(const array_ref& other) const {
    return !operator!=(other);
  }

  const array_ref<T, Shape>& ref() const {
    return *this;
  }
  const array_ref<const T, Shape> cref() const {
    return array_ref<const T, Shape>(base_, shape_);
  }

  /** Allow conversion from array_ref<T> to array_ref<const T>. */
  operator array_ref<const T, Shape>() const { return cref(); }

  /** Change the shape of the array to 'new_shape', and move the base pointer by
   * 'offset'. */
  void set_shape(Shape new_shape, index_t offset = 0) {
    shape_ = std::move(new_shape);
    base_ = internal::pointer_add(base_, offset);
  }
};

/** array_ref with an arbitrary shape of the compile-time constant 'Rank'. */
template <typename T, size_t Rank>
using array_ref_of_rank = array_ref<T, shape_of_rank<Rank>>;

/** array_ref with a 'dense_dim' innermost dimension, and an arbitrary shape
 * otherwise, of the compile-time constant 'Rank'. */
template <typename T, size_t Rank>
using dense_array_ref = array_ref<T, dense_shape<Rank>>;

/** A multi-dimensional array container that owns an allocation of memory. This
 * container is designed to mirror the semantics of std::vector where possible.
 */
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
class array {
 public:
  /** Type of the values stored in this array. */
  typedef T value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef typename std::allocator_traits<Alloc>::pointer pointer;
  typedef typename std::allocator_traits<Alloc>::const_pointer const_pointer;
  /** Type of the shape of this array. */
  typedef Shape shape_type;
  typedef typename Shape::index_type index_type;
  typedef size_t size_type;
  /** Type of the allocator used to allocate memory in this array. */
  typedef Alloc allocator_type;

 private:
  Alloc alloc_;
  pointer buffer_;
  size_t buffer_size_;
  pointer base_;
  Shape shape_;

  // After allocate the array is allocated but uninitialized.
  void allocate() {
    assert(!buffer_);
    size_t flat_extent = shape_.flat_extent();
    if (flat_extent > 0) {
      buffer_size_ = flat_extent;
      buffer_ = std::allocator_traits<Alloc>::allocate(alloc_, buffer_size_);
    }
    base_ = buffer_ - shape_.flat_min();
  }

  // Call the constructor on all of the elements of the array.
  void construct() {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) {
      std::allocator_traits<Alloc>::construct(alloc_, &x);
    });
  }
  void construct(const T& init) {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) {
      std::allocator_traits<Alloc>::construct(alloc_, &x, init);
    });
  }
  void copy_construct(const array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape());
    copy_shape_traits<Shape>::for_each_value(other.shape(), other.base(), shape_, base_,
                                             [&](const value_type& src, value_type& dest) {
      std::allocator_traits<Alloc>::construct(alloc_, &dest, src);
    });
  }
  void move_construct(array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape());
    copy_shape_traits<Shape>::for_each_value(other.shape(), other.base(), shape_, base_,
                                             [&](value_type& src, value_type& dest) {
      std::allocator_traits<Alloc>::construct(alloc_, &dest, std::move(src));
    });
  }

  // Call the destructor on every element.
  void destroy() {
    assert(base_ || shape_.empty());
    for_each_value([&](T& x) {
      std::allocator_traits<Alloc>::destroy(alloc_, &x);
    });
  }

  void deallocate() {
    if (base_) {
      destroy();
      base_ = nullptr;
      shape_ = Shape();
      std::allocator_traits<Alloc>::deallocate(alloc_, buffer_, buffer_size_);
      buffer_ = nullptr;
    }
  }

 public:
  /** Construct an array with a default constructed Shape. Most shapes by
   * default are empty, but a Shape with non-zero compile-time constants for all
   * extents will be non-empty. */
  array() : array(Shape()) {}
  explicit array(const Alloc& alloc) : array(Shape(), alloc) {}
  /** Construct an array with a particular 'shape', allocated by 'alloc'. All
   * elements in the array are copy-constructed from 'value'. */
  array(Shape shape, const T& value, const Alloc& alloc = Alloc()) : array(alloc) {
    assign(std::move(shape), value);
  }
  /** Construct an array with a particular 'shape', allocated by 'alloc', with
   * default constructed elements. */
  explicit array(Shape shape, const Alloc& alloc = Alloc())
      : alloc_(alloc), buffer_(nullptr), buffer_size_(0), base_(nullptr), shape_(std::move(shape)) {
    allocate();
    construct();
  }
  /** Copy construct from another array 'other', using copy's allocator. This is
   * a deep copy of the contents of 'other'. */
  array(const array& other)
      : array(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.get_allocator())) {
    assign(other);
  }
  /** Copy construct from another array 'other'. The array is allocated using
   * 'alloc'. This is a deep copy of the contents of 'other'. */
  array(const array& other, const Alloc& alloc) : array(alloc) {
    assign(other);
  }
  /** Move construct from another array 'other'. If the allocator of this array
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

  /** Assign the contents of the array as a copy of 'other'. The array is
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

    if (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
      if (alloc_ != other.get_allocator()) {
        deallocate();
      }
      alloc_ = other.get_allocator();
    }

    assign(other);
    return *this;
  }
  /** Assign the contents of the array by moving from 'other'. If the allocator
   * can be propagated on move assignment, the allocation of 'other' is moved in
   * an O(1) operation. If the allocator cannot be propagated, each element is
   * move-assigned from 'other'. */
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

  /** Assign the contents of the array to be a copy or move of 'other'. The
   * array is destroyed, reallocated if necessary, and then each element is
   * copy- or move-constructed from 'other'. */
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

  /** Assign the contents of this array to have 'shape' with each element copy
   * constructed from 'value'. */
  void assign(Shape shape, const T& value) {
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
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  reference operator() (Indices... indices) { return base_[shape_(indices...)]; }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  const_reference operator() (Indices... indices) const { return base_[shape_(indices...)]; }

  /** Call a function with a reference to each value in this array. The order in
   * which 'fn' is called is undefined. */
  template <typename Fn>
  void for_each_value(Fn&& fn) {
    shape_traits<Shape>::for_each_value(shape_, base_, fn);
  }
  template <typename Fn>
  void for_each_value(Fn&& fn) const {
    shape_traits<Shape>::for_each_value(shape_, base_, fn);
  }

  /** Pointer to the element at the min index of the shape. */
  pointer base() { return base_; }
  const_pointer base() const { return base_; }

  /** Pointer to the element at the beginning of the flat array. */
  pointer data() { return internal::pointer_add(base_, shape_.flat_min()); }
  const_pointer data() const { return internal::pointer_add(base_, shape_.flat_min()); }

  /** Shape of this array. */
  const Shape& shape() const { return shape_; }

  static constexpr size_t rank() { return Shape::rank(); }
  static constexpr bool is_scalar() { return Shape::is_scalar(); }
  template <size_t D>
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
  void reshape(Shape new_shape) {
    // Allocate an array with the new shape.
    array<T, Shape, Alloc> new_array(new_shape);

    // Move the common elements to the new array.
    Shape intersection = intersect(shape_, new_shape);
    copy_shape_traits<Shape>::for_each_value(shape_, base_, intersection, new_array.base(),
                                             [](T& src, T& dest) {
      dest = std::move(src);
    });

    // Swap this with the new array.
    swap(new_array);
  }

  /** Change the shape of the array to 'new_shape', and move the base pointer by
   * 'offset'. */
  void set_shape(Shape new_shape, index_t offset = 0) {
    assert(new_shape.is_subset_of(shape(), -offset));
    shape_ = std::move(new_shape);
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

  /** Compare the contents of this array to 'other'. For two arrays to be
   * considered equal, they must have the same shape, and all elements
   * addressable by the shape must also be equal. */
  bool operator!=(const array& other) const { return cref() != other.cref(); }
  bool operator==(const array& other) const { return cref() == other.cref(); }

  /** Swap the contents of two arrays. This performs zero copies or moves of
   * individual elements. */
  void swap(array& other) {
    using std::swap;

    if (std::allocator_traits<Alloc>::propagate_on_container_swap::value) {
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

/** An array type with an arbitrary shape of rank 'Rank'. */
template <typename T, size_t Rank, typename Alloc = std::allocator<T>>
using array_of_rank = array<T, shape_of_rank<Rank>, Alloc>;

/** array with a 'dense_dim' innermost dimension, and an arbitrary shape
 * otherwise, of rank 'Rank'. */
template <typename T, size_t Rank, typename Alloc = std::allocator<T>>
using dense_array = array<T, dense_shape<Rank>, Alloc>;

/** Make a new array with shape 'shape', allocated using 'alloc'. */
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
auto make_array(const Shape& shape, const Alloc& alloc = Alloc()) {
  return array<T, Shape, Alloc>(shape, alloc);
}
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
auto make_array(const Shape& shape, const T& value, const Alloc& alloc = Alloc()) {
  return array<T, Shape, Alloc>(shape, value, alloc);
}

/** Swap the contents of two arrays. */
template <typename T, typename Shape, typename Alloc>
void swap(array<T, Shape, Alloc>& a, array<T, Shape, Alloc>& b) {
  a.swap(b);
}

/** Copy the contents of the 'src' array or array_ref to the 'dest' array or
 * array_ref. The range of the shape of 'dest' will be copied, and must be in
 * bounds of 'src'. */
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest>
void copy(const array_ref<TSrc, ShapeSrc>& src, const array_ref<TDest, ShapeDest>& dest) {
  if (dest.shape().empty()) {
    return;
  }
  if (!src.shape().is_in_range(dest.shape().min()) ||
      !src.shape().is_in_range(dest.shape().max())) {
    NDARRAY_THROW_OUT_OF_RANGE("dest indices out of range of src");
  }

  copy_shape_traits<ShapeSrc, ShapeDest>::for_each_value(src.shape(), src.base(), dest.shape(), dest.base(),
                                                         [](const TSrc& src_i, TDest& dest_i) {
    dest_i = src_i;
  });
}
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest, typename AllocDest>
void copy(const array_ref<TSrc, ShapeSrc>& src, array<TDest, ShapeDest, AllocDest>& dest) {
  copy(src, dest.ref());
}
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest, typename AllocSrc>
void copy(const array<TSrc, ShapeSrc, AllocSrc>& src, const array_ref<TDest, ShapeDest>& dest) {
  copy(src.cref(), dest);
}
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest, typename AllocSrc, typename AllocDest>
void copy(const array<TSrc, ShapeSrc, AllocSrc>& src, array<TDest, ShapeDest, AllocDest>& dest) {
  copy(src.cref(), dest.ref());
}

/** Make a copy of the 'src' array or array_ref with a new shape 'shape'. */
template <typename T, typename ShapeSrc, typename ShapeDest,
  typename Alloc = std::allocator<typename std::remove_const<T>::type>>
auto make_copy(const array_ref<T, ShapeSrc>& src, const ShapeDest& shape,
               const Alloc& alloc = Alloc()) {
  array<typename std::remove_const<T>::type, ShapeDest, Alloc> dest(shape, alloc);
  copy(src, dest);
  return dest;
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc,
  typename AllocDest = AllocSrc>
auto make_copy(const array<T, ShapeSrc, AllocSrc>& src, const ShapeDest& shape,
               const AllocDest& alloc = AllocDest()) {
  return make_copy(src.cref(), shape, alloc);
}

/** Make a copy of the 'src' array or array_ref with a dense shape of the same
 * rank as 'src'. */
template <typename T, typename ShapeSrc,
  typename Alloc = std::allocator<typename std::remove_const<T>::type>>
auto make_dense_copy(const array_ref<T, ShapeSrc>& src,
                     const Alloc& alloc = Alloc()) {
  return make_copy(src, make_dense(src.shape()), alloc);
}
template <typename T, typename ShapeSrc, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_dense_copy(const array<T, ShapeSrc, AllocSrc>& src,
                     const AllocDest& alloc = AllocDest()) {
  return make_dense_copy(src.cref(), alloc);
}

/** Make a copy of the 'src' array or array_ref with a compact version of 'src's
 * shape. */
template <typename T, typename Shape,
  typename Alloc = std::allocator<typename std::remove_const<T>::type>>
auto make_compact_copy(const array_ref<T, Shape>& src,
                       const Alloc& alloc = Alloc()) {
  return make_copy(src, make_compact(src.shape()), alloc);
}
template <typename T, typename Shape, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_compact_copy(const array<T, Shape, AllocSrc>& src,
                       const AllocDest& alloc = AllocDest()) {
  return make_compact_copy(src.cref(), alloc);
}

/** Move the contents from the 'src' array or array_ref to the 'dest' array or
 * array_ref. The range of the shape of 'dest' will be moved, and must be in
 * bounds of 'src'. */
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest>
void move(const array_ref<TSrc, ShapeSrc>& src, const array_ref<TDest, ShapeDest>& dest) {
  if (dest.shape().empty()) {
    return;
  }
  if (!src.shape().is_in_range(dest.shape().min()) ||
      !src.shape().is_in_range(dest.shape().max())) {
    NDARRAY_THROW_OUT_OF_RANGE("dest indices out of range of src");
  }

  copy_shape_traits<ShapeSrc, ShapeDest>::for_each_value(src.shape(), src.base(), dest.shape(), dest.base(),
                                                         [](TSrc& src_i, TDest& dest_i) {
    dest_i = std::move(src_i);
  });
}
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest, typename AllocDest>
void move(const array_ref<TSrc, ShapeSrc>& src, array<TDest, ShapeDest, AllocDest>& dest) {
  move(src, dest.ref());
}
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest, typename AllocSrc>
void move(array<TSrc, ShapeSrc, AllocSrc>& src, const array_ref<TDest, ShapeDest>& dest) {
  move(src.ref(), dest);
}
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest, typename AllocSrc, typename AllocDest>
void move(array<TSrc, ShapeSrc, AllocSrc>& src, array<TDest, ShapeDest, AllocDest>& dest) {
  move(src.ref(), dest.ref());
}

/** Make a copy of the 'src' array or array_ref with a new shape 'shape'. The
 * elements of 'src' are moved to the result. */
template <typename T, typename ShapeSrc, typename ShapeDest, typename Alloc = std::allocator<T>>
auto make_move(const array_ref<T, ShapeSrc>& src, const ShapeDest& shape,
               const Alloc& alloc = Alloc()) {
  array<T, ShapeDest, Alloc> dest(shape, alloc);
  move(src, dest);
  return dest;
}
// TODO: Should this taken an rvalue reference for src, and should it move the
// whole array if the shapes are equal?
// (https://github.com/dsharlet/array/issues/8)
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc,
  typename AllocDest = AllocSrc>
auto make_move(array<T, ShapeSrc, AllocSrc>& src, const ShapeDest& shape,
               const AllocDest& alloc = AllocDest()) {
  return make_move(src.ref(), shape, alloc);
}

/** Make a copy of the 'src' array or array_ref with a dense shape of the same
 * rank as 'src'. The elements of 'src' are moved to the result. */
template <typename T, typename ShapeSrc, typename Alloc = std::allocator<T>>
auto make_dense_move(const array_ref<T, ShapeSrc>& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_dense(src.shape()), alloc);
}
// TODO: Should this taken an rvalue reference for src, and should it move the
// whole array if the shapes are equal?
// (https://github.com/dsharlet/array/issues/8)
template <typename T, typename ShapeSrc, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_dense_move(array<T, ShapeSrc, AllocSrc>& src, const AllocDest& alloc = AllocDest()) {
  return make_dense_move(src.ref(), alloc);
}

/** Make a copy of the 'src' array or array_ref with a compact version of 'src's
 * shape. The elements of 'src' are moved to the result. */
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
auto make_compact_move(const array_ref<T, Shape>& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_compact(src.shape()), alloc);
}
// TODO: Should this taken an rvalue reference for src, and should it move the
// whole array if the shapes are equal?
// (https://github.com/dsharlet/array/issues/8)
template <typename T, typename Shape, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_compact_move(array<T, Shape, AllocSrc>& src, const AllocDest& alloc = AllocDest()) {
  return make_compact_move(src.ref(), alloc);
}

/** Reinterpret the array or array_ref 'a' of type 'T' to have a different type
 * 'U'. The size of 'T' must be equal to the size of 'U'. */
template <typename U, typename T, typename Shape>
array_ref<U, Shape> reinterpret(const array_ref<T, Shape>& a) {
  static_assert(sizeof(T) == sizeof(U), "sizeof(reinterpreted type U) != sizeof(array type T)");
  return array_ref<U, Shape>(reinterpret_cast<U*>(a.base()), a.shape());
}
template <typename U, typename T, typename Shape, typename Alloc>
array_ref<U, Shape> reinterpret(array<T, Shape, Alloc>& a) {
  return reinterpret<U>(a.ref());
}
template <typename U, typename T, typename Shape, typename Alloc>
array_ref<const U, Shape> reinterpret(const array<T, Shape, Alloc>& a) {
  return reinterpret<const U>(a.cref());
}

/** Reinterpret the shape of the array or array_ref 'a' to be a new shape
 * 'new_shape', with a base pointer offset 'offset'. */
template <typename NewShape, typename T, typename OldShape>
array_ref<T, NewShape> reinterpret_shape(const array_ref<T, OldShape>& a,
                                         NewShape new_shape, index_t offset = 0) {
  assert(new_shape.is_subset_of(a.shape(), -offset));
  return array_ref<T, NewShape>(a.base() + offset, std::move(new_shape));
}
template <typename NewShape, typename T, typename OldShape, typename Allocator>
array_ref<T, NewShape> reinterpret_shape(array<T, OldShape, Allocator>& a,
                                         NewShape new_shape, index_t offset = 0) {
  return reinterpret_shape(a.ref(), new_shape, offset);
}
template <typename NewShape, typename T, typename OldShape, typename Allocator>
array_ref<const T, NewShape> reinterpret_shape(const array<T, OldShape, Allocator>& a,
                                               NewShape new_shape, index_t offset = 0) {
  return reinterpret_shape(a.cref(), new_shape, offset);
}

/** std::allocator-compatible Allocator that owns a buffer of fixed size, which
 * will be placed on the stack if the owning container is allocated on the
 * stack. This can only be used with containers that have a maximum of one
 * concurrent live allocation, which is the case for array::array. */
// TODO: "stack_allocator" isn't a good name for this. It's a fixed allocation,
// but not necessarily a stack allocation
// (https://github.com/dsharlet/array/issues/6).
template <class T, size_t N>
class stack_allocator {
  alignas(T) char buffer[N * sizeof(T)];
  bool allocated;

 public:
  typedef T value_type;

  typedef std::false_type propagate_on_container_copy_assignment;
  typedef std::false_type propagate_on_container_move_assignment;
  typedef std::false_type propagate_on_container_swap;

  static stack_allocator select_on_container_copy_construction(const stack_allocator&) {
    return stack_allocator();
  }

  stack_allocator() : allocated(false) {}
  template <class U, size_t U_N> constexpr
  stack_allocator(const stack_allocator<U, U_N>&) noexcept : allocated(false) {}
  // TODO: Most of these constructors/assignment operators are hacks,
  // because the C++ STL I'm using seems to not be respecting the
  // propagate typedefs or the 'select_on_...' function above.
  stack_allocator(const stack_allocator&) noexcept : allocated(false) {}
  stack_allocator(stack_allocator&&) noexcept : allocated(false) {}
  stack_allocator& operator=(const stack_allocator&) { return *this; }
  stack_allocator& operator=(stack_allocator&&) { return *this; }

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
  friend bool operator==(const stack_allocator<T, N>& a, const stack_allocator<U, U_N>& b) {
    return &a.buffer[0] == &b.buffer[0];
  }

  template <class U, size_t U_N>
  friend bool operator!=(const stack_allocator<T, N>& a, const stack_allocator<U, U_N>& b) {
    return &a.buffer[0] != &b.buffer[0];
  }
};

}  // namespace nda

#endif  // NDARRAY_ARRAY_H

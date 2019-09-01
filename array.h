#ifndef ARRAY_ARRAY_H
#define ARRAY_ARRAY_H

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>

namespace array {

typedef std::size_t size_t;
typedef std::ptrdiff_t index_t;

enum : index_t {
  // TODO: It would be best to use a more unreasonable value for
  // this (strides of -1 might be useful), but it makes error messages
  // absurd.
  UNK = -1, //std::numeric_limits<index_t>::min(),
};

namespace internal {

// Given a compile-time static value, reconcile a compile-time
// static value and runtime value.
template <index_t Value>
index_t reconcile(index_t value) {
  if (Value != UNK) {
    // It would be nice to assert here that Value == value. But, this
    // is used in the innermost loops, so when asserts are on, this
    // ruins performance. It is also a less helpful place to catch
    // errors like this, because the bug it is catching is caused by
    // an issue much earlier than this, so it is better to assert
    // there instead.
    return Value;
  } else {
    return value;
  }
}

// Signed integer division in C/C++ is terrible. These implementations
// of Euclidean division and mod are taken from:
// https://github.com/halide/Halide/blob/1a0552bb6101273a0e007782c07e8dafe9bc5366/src/CodeGen_Internal.cpp#L358-L408
template <typename T>
T euclidean_div(T a, T b) {
  T q = a / b;
  T r = a - q * b;
  T bs = b >> (sizeof(T) * 8 - 1);
  T rs = r >> (sizeof(T) * 8 - 1);
  return q - (rs & bs) + (rs & ~bs);
}

template <typename T>
T euclidean_mod(T a, T b) {
  T r = a % b;
  T sign_mask = r >> (sizeof(T) * 8 - 1);
  return r + (sign_mask & std::abs(b));
}

}  // namespace internal

/** An iterator over a range of indices. */
class dim_iterator {
  index_t i;

 public:
  dim_iterator(index_t i) : i(i) {}

  bool operator==(const dim_iterator& r) const { return i == r.i; }
  bool operator!=(const dim_iterator& r) const { return i != r.i; }

  index_t operator *() const { return i; }

  dim_iterator operator++(int) { return dim_iterator(i++); }
  dim_iterator& operator++() { ++i; return *this; }
};

/** Describes one dimension of an array. The template parameters
 * enable providing compile time constants for the min, extent, and
 * stride of the dim. */
template <index_t Min = UNK, index_t Extent = UNK, index_t Stride = UNK>
class dim {
 protected:
  index_t min_;
  index_t extent_;
  index_t stride_;

 public:
  /** Construct a new dim object. If the Min, Extent, or Stride
   * compile-time template parameters are not 'UNK', these runtime
   * values must match the compile-time values. */
  dim(index_t min, index_t extent, index_t stride = Stride)
    : min_(min), extent_(extent), stride_(stride) {
    assert(min == Min || Min == UNK);
    assert(extent == Extent || Extent == UNK);
    assert(stride == Stride || Stride == UNK);
  }
  dim(index_t extent = Extent) : dim(0, extent) {}
  dim(const dim&) = default;
  dim(dim&&) = default;
  /** Copy a dim object, possibly with different compile-time template
   * parameters. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride>
  dim(const dim<CopyMin, CopyExtent, CopyStride>& copy)
      : dim(copy.min(), copy.extent(), copy.stride()) {}

  dim& operator=(const dim&) = default;
  dim& operator=(dim&&) = default;
  /** Copy assignment of a dim object, possibly with different
   * compile-time template parameters. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride>
  dim& operator=(const dim<CopyMin, CopyExtent, CopyStride>& copy) {
    assert(copy.min() == Min || Min == UNK);
    min_ = copy.min();
    assert(copy.extent() == Extent || Extent == UNK);
    extent_ = copy.extent();
    assert(copy.stride() == Stride || Stride == UNK);
    stride_ = copy.stride();
  }

  /** Index of the first element in this dim. */
  index_t min() const { return internal::reconcile<Min>(min_); }
  void set_min(index_t min) {
    assert(min == Min || Min == UNK);
    min_ = min;
  }
  /** Number of elements in this dim. */
  index_t extent() const { return internal::reconcile<Extent>(extent_); }
  void set_extent(index_t extent) {
    assert(extent == Extent || Extent == UNK);
    extent_ = extent;
  }
  /** Distance betwen elements in this dim. */
  index_t stride() const { return internal::reconcile<Stride>(stride_); }
  void set_stride(index_t stride) {
    assert(stride == Stride || Stride == UNK);
    stride_ = stride;
  }
  /** Index of the last element in this dim. */
  index_t max() const { return min() + extent() - 1; }

  /** Offset in memory of an index in this dim. */
  index_t offset(index_t at) const { return (at - min()) * stride(); }

  /** Returns true if 'at' is within the range [min(), max()]. */
  bool is_in_range(index_t at) const { return min() <= at && at <= max(); }

  /** Make an iterator referring to the first element in this dim. */
  dim_iterator begin() const { return dim_iterator(min()); }
  /** Make an iterator referring to one past the last element in this
   * dim. */
  dim_iterator end() const { return dim_iterator(max() + 1); }

  /** dim objects are considered equal if their min, extent, and
   * strides are equal. */
  bool operator==(const dim& other) const {
    return min() == other.min() && extent() == other.extent() && stride() == other.stride();
  }
  bool operator!=(const dim& other) const {
    return min() != other.min() || extent() != other.extent() || stride() != other.stride();
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

/** A specialization of dim where the stride is known to be one. */
template <index_t Min = UNK, index_t Extent = UNK>
using dense_dim = dim<Min, Extent, 1>;

/** A specialization of dim where the stride is known to be zero. */
template <index_t Min = UNK, index_t Extent = UNK>
using broadcast_dim = dim<Min, Extent, 0>;

/** A dim where the indices wrap around outside the range of the dim. */
template <index_t Extent = UNK, index_t Stride = UNK>
class folded_dim {
 protected:
  index_t extent_;
  index_t stride_;
  
 public:
  /** Construct a new dim object. If the Extent or Stride compile-time
   * template parameters are not 'UNK', these runtime values must
   * match the compile-time values. */
  folded_dim(index_t extent = Extent, index_t stride = Stride)
    : extent_(extent), stride_(stride) {
    assert(extent == Extent || Extent == UNK);
    assert(stride == Stride || Stride == UNK);
  }
  folded_dim(const folded_dim&) = default;
  folded_dim(folded_dim&&) = default;

  folded_dim& operator=(const folded_dim&) = default;
  folded_dim& operator=(folded_dim&&) = default;

  /** Non-folded range of the dim. */
  index_t extent() const { return internal::reconcile<Extent>(extent_); }
  void set_extent(index_t extent) {
    assert(extent == Extent || Extent == UNK);
    extent_ = extent;
  }
  /** Distance in memory between indices of this dim. */
  index_t stride() const { return internal::reconcile<Stride>(stride_); }
  void set_stride(index_t stride) {
    assert(stride == Stride || Stride == UNK);
    stride_ = stride;
  }

  /** In a folded dim, the min and max are unbounded. */
  index_t min() const { return 0; }
  index_t max() const { return extent() - 1; }

  /** Make an iterator referring to the first element in this dim. */
  dim_iterator begin() const { return dim_iterator(min()); }
  /** Make an iterator referring to one past the last element in this dim. */
  dim_iterator end() const { return dim_iterator(max() + 1); }

  /** Offset in memory of an element in this dim. */
  index_t offset(index_t at) const {
    return internal::euclidean_mod(at, extent()) * stride();
  }

  /** In a folded dim, all indices are in range. */
  bool is_in_range(index_t at) const { return true; }

  /** dim objects are considered equal if their min, extent, and
   * strides are equal. */
  bool operator==(const folded_dim& other) const {
    return extent() == other.extent() && stride() == other.stride();
  }
  bool operator!=(const folded_dim& other) const {
    return extent() != other.extent() || stride() != other.stride();
  }
};

template <index_t Extent, index_t Stride>
dim_iterator begin(const folded_dim<Extent, Stride>& d) {
  return d.begin();
}
template <index_t Extent, index_t Stride>
dim_iterator end(const folded_dim<Extent, Stride>& d) {
  return d.end();
}

/** Clamp an index to the range [min, max]. */
inline index_t clamp(index_t x, index_t min, index_t max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

/** Clamp an index to the range described by a dim. */
template <typename Dim>
inline index_t clamp(index_t x, const Dim& d) {
  return clamp(x, d.min(), d.max());
}

namespace internal {

// Base and general case for the sum of a variadic list of values.
inline index_t sum() {
  return 0;
}
inline index_t product() {
  return 1;
}
inline index_t variadic_max() {
  return std::numeric_limits<index_t>::min();
}

template <typename... Rest>
index_t sum(index_t first, Rest... rest) {
  return first + sum(rest...);
}
template <typename... Rest>
index_t product(index_t first, Rest... rest) {
  return first * product(rest...);
}
template <typename... Rest>
index_t variadic_max(index_t first, Rest... rest) {
  return std::max(first, variadic_max(rest...));
}

// Computes the product of the extents of the dims.
template <typename Dims, size_t... Is>
index_t product_of_extents_impl(const Dims& dims, std::index_sequence<Is...>) {
  return product(std::get<Is>(dims).extent()...);
}

template <typename Dims>
index_t product_of_extents(const Dims& dims) {
  constexpr size_t dims_rank = std::tuple_size<Dims>::value;
  return product_of_extents_impl(dims, std::make_index_sequence<dims_rank>());
}

// Computes the sum of the offsets of a list of dims and indices.
template <typename Dims, typename Indices, size_t... Is>
index_t flat_offset_impl(const Dims& dims, const Indices& indices, std::index_sequence<Is...>) {
  return sum(std::get<Is>(dims).offset(std::get<Is>(indices))...);
}

template <typename Dims, typename Indices>
index_t flat_offset(const Dims& dims, const Indices& indices) {
  constexpr size_t dims_rank = std::tuple_size<Dims>::value;
  constexpr size_t indices_rank = std::tuple_size<Indices>::value;
  static_assert(dims_rank == indices_rank, "dims and indices must have the same rank.");
  return flat_offset_impl(dims, indices, std::make_index_sequence<dims_rank>());
}

// Computes one more than the sum of the offsets of the last index in every dim.
template <typename Dims, size_t... Is>
index_t flat_extent_impl(const Dims& dims, std::index_sequence<Is...>) {
  return 1 + sum((std::get<Is>(dims).extent() - 1) * std::get<Is>(dims).stride()...);
}

template <typename Dims>
index_t flat_extent(const Dims& dims) {
  constexpr size_t rank = std::tuple_size<Dims>::value;
  return flat_extent_impl(dims, std::make_index_sequence<rank>());
}

// Checks if all indices are in range of each corresponding dim.
template <typename Dims, typename Indices, size_t... Is>
index_t is_in_range_impl(const Dims& dims, const Indices& indices, std::index_sequence<Is...>) {
  return sum((std::get<Is>(dims).is_in_range(std::get<Is>(indices)) ? 0 : 1)...) == 0;
}

template <typename Dims, typename Indices>
bool is_in_range(const Dims& dims, const Indices& indices) {
  constexpr size_t dims_rank = std::tuple_size<Dims>::value;
  constexpr size_t indices_rank = std::tuple_size<Indices>::value;
  static_assert(dims_rank == indices_rank, "dims and indices must have the same rank.");
  return is_in_range_impl(dims, indices, std::make_index_sequence<dims_rank>());
}

template <typename Dims, size_t... Is>
index_t max_stride(const Dims& dims, std::index_sequence<Is...>) {
  return variadic_max(std::get<Is>(dims).stride() * std::get<Is>(dims).extent()...);
}

// Get a tuple of all of the mins of the shape.
template <typename Shape, size_t... Is>
auto mins(const Shape& s, std::index_sequence<Is...>) {
  return std::make_tuple(s.template dim<Is>().min()...);
}

template <typename Shape>
auto mins(const Shape& s) {
  return mins(s, std::make_index_sequence<Shape::rank()>());
}

// Get a tuple of all of the extents of the shape.
template <typename Shape, size_t... Is>
auto extents(const Shape& s, std::index_sequence<Is...>) {
  return std::make_tuple(s.template dim<Is>().extent()...);
}

template <typename Shape>
auto extents(const Shape& s) {
  return extents(s, std::make_index_sequence<Shape::rank()>());
}

// Get a tuple of all of the maxes of the shape.
template <typename Shape, size_t... Is>
auto maxes(const Shape& s, std::index_sequence<Is...>) {
  return std::make_tuple(s.template dim<Is>().max()...);
}

template <typename Shape>
auto maxes(const Shape& s) {
  return maxes(s, std::make_index_sequence<Shape::rank()>());
}

// Resolve unknown dim quantities. Unknown extents become zero, and
// unknown strides are replaced with increasing strides.
inline void resolve_unknowns_impl(index_t current_stride) {}

template <typename Dim0, typename... Dims>
void resolve_unknowns_impl(index_t current_stride, Dim0& dim0, Dims&... dims) {
  if (dim0.extent() == UNK) {
    dim0.set_extent(0);
  }
  if (dim0.stride() == UNK) {
    dim0.set_stride(current_stride);
    current_stride *= dim0.extent();
  }
  resolve_unknowns_impl(current_stride, std::forward<Dims&>(dims)...);
}

template <typename Dims, size_t... Is>
void resolve_unknowns_impl(index_t current_stride, Dims& dims, std::index_sequence<Is...>) {
  resolve_unknowns_impl(current_stride, std::get<Is>(dims)...);
}

template <typename Dims>
void resolve_unknowns(Dims& dims) {
  constexpr size_t rank = std::tuple_size<Dims>::value;
  index_t known_stride = max_stride(dims, std::make_index_sequence<rank>());
  index_t current_stride = std::max(static_cast<index_t>(1), known_stride);

  resolve_unknowns_impl(current_stride, dims, std::make_index_sequence<rank>());
}

// A helper to transform an array to a tuple.
template <typename T, size_t... Is>
auto array_to_tuple_impl(const std::array<T, sizeof...(Is)>& a, std::index_sequence<Is...>) {
  return std::make_tuple(a[Is]...);
}

template <typename T, size_t N>
auto array_to_tuple(const std::array<T, N>& a) {
  return array_to_tuple_impl(a, std::make_index_sequence<N>());
}

template <typename T, size_t N>
auto default_array_to_tuple() {
  return array_to_tuple(std::array<T, N>());
}

template <typename... New, typename... Old, size_t... Is>
std::tuple<New...> convert_tuple_impl(const std::tuple<Old...>& in, std::index_sequence<Is...>) {
  return std::tuple<New...>(std::get<Is>(in)...);
}

template <typename... New, typename... Old>
std::tuple<New...> convert_tuple(const std::tuple<Old...>& in) {
  static_assert(sizeof...(New) == sizeof...(Old), "tuple conversion of differently sized tuples");
  return convert_tuple_impl<New...>(in, std::make_index_sequence<sizeof...(Old)>());
}

}  // namespace internal

/** A list of dims describing a multi-dimensional space of
 * indices. The first dim is considered the 'innermost' dimension,
 * and the last dim is the 'outermost' dimension. */
template <typename... Dims>
class shape {
  std::tuple<Dims...> dims_;

 public:
  /** When constructing shapes, unknown extents are set to 0, and
   * unknown strides are set to the currently largest known
   * stride. This is done in innermost-to-outermost order. */
  shape() { internal::resolve_unknowns(dims_); }
  shape(std::tuple<Dims...> dims) : dims_(std::move(dims)) { internal::resolve_unknowns(dims_); }
  shape(Dims... dims) : dims_(std::forward<Dims>(dims)...) { internal::resolve_unknowns(dims_); }
  shape(const shape&) = default;
  shape(shape&&) = default;
  /** Construct this shape from a different kind of shape. */
  template <typename... OtherDims>
  shape(const shape<OtherDims...>& conversion)
    : dims_(internal::convert_tuple<Dims...>(conversion.dims())) {}

  shape& operator=(const shape&) = default;
  shape& operator=(shape&&) = default;

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
  typedef decltype(internal::default_array_to_tuple<index_t, rank()>()) index_type;

  /** Check if a list of indices is in range of this shape. */
  template <typename... Indices>
  bool is_in_range(const std::tuple<Indices...>& indices) const {
    return internal::is_in_range(dims_, indices);
  }
  template <typename... Indices>
  bool is_in_range(Indices... indices) const {
    return is_in_range(std::make_tuple<Indices...>(std::forward<Indices>(indices)...));
  }

  /** Check to see if another shape is in range of this shape. A shape
   * is in range of another shape if the min and max of each dimension
   * is in range of the other shape. */
  template <typename... OtherDims>
  bool is_shape_in_range(const shape<OtherDims...>& other_shape) const {
    return is_in_range(internal::mins(other_shape)) && is_in_range(internal::maxes(other_shape));
  }

  /** Compute the flat offset of the indices. */
  template <typename... Indices>
  index_t at(const std::tuple<Indices...>& indices) const {
    return internal::flat_offset(dims_, indices);
  }
  template <typename... Indices>
  index_t at(Indices... indices) const {
    return at(std::make_tuple<Indices...>(std::forward<Indices>(indices)...));
  }
  template <typename... Indices>
  index_t operator() (const std::tuple<Indices...>& indices) const {
    return at(indices);
  }
  template <typename... Indices>
  index_t operator() (Indices... indices) const {
    return at(std::make_tuple<Indices...>(std::forward<Indices>(indices)...));
  }
  
  /** Get a specific dim of this shape. */
  template <size_t D>
  const auto& dim() const { return std::get<D>(dims_); }

  /** Get a tuple of the dims of this shape. */
  std::tuple<Dims...>& dims() { return dims_; }
  const std::tuple<Dims...>& dims() const { return dims_; }

  /** Compute the flat extent of this shape. This is the extent of the
   * valid range of values returned by at or operator(). */
  index_t flat_extent() const { return internal::flat_extent(dims_); }

  /** Compute the total number of items in the shape. */
  index_t size() const { return internal::product_of_extents(dims_); }

  /** A shape is empty if its size is 0. */
  bool empty() const { return size() == 0; }

  /** Returns true if this shape is an injective function mapping
   * indices to flat indices. If the dims overlap, or a dim has stride
   * zero, multiple indices will map to the same flat index. */
  bool is_one_to_one() const {
    // We need to solve:
    //
    //   x0*S0 + x1*S1 + x2*S2 + ... == y0*S0 + y1*S1 + y2*S2 + ...
    //
    // where xN, yN are (different) indices, and SN are the strides of
    // this shape. This is equivalent to:
    //
    //   (x0 - y0)*S0 + (x1 - y1)*S1 + ... == 0
    //
    // We don't actually care what xi and yi are, so this is equivalent
    // to:
    //
    //   x0*S0 + x1*S1 + x2*S2 + ... == 0
    //
    // where xi != 0. This is a linear diophantine equation, and we
    // already have one solution at xi = 0, so we just need to find
    // other solutions, and check that they are in range.

    // TODO: This is pretty hard. I think we need to rewrite the
    // equation as a system of linear diophantine equations, and
    // then use the "Hermite normal form" to get the unbounded
    // solutions, and then do some combinatoric search for the
    // in-bounds solutions. This is an NP-hard problem, but the
    // size of the problems are small, and I don't think these
    // functions need to be fast.
    return true;
  }

  /** Returns true if this shape projects to a set of flat indices
   * that is a subset of the other shape's projection to flat
   * indices. */
  bool is_subset_of(const shape& other) const {
    // TODO: This is also hard, maybe even harder than is_one_to_one.
    return true;
  }

  /** Returns true if this shape is dense in memory. A shape is
   * 'dense' if there are no unaddressable flat indices between the
   * first and last addressable flat elements. */
  bool is_dense() const { return size() == flat_extent(); }

  /** A shape is equal to another shape if all of the dimensions are
   * equal. */
  bool operator==(const shape& other) const { return dims_ == other.dims_; }
  bool operator!=(const shape& other) const { return dims_ != other.dims_; }
};

// TODO: Try to avoid needing this specialization. The only reason
// it is necessary is because the above defines two default constructors.
template <>
class shape<> {
 public:
  shape() {}

  constexpr size_t rank() const { return 0; }

  typedef std::tuple<> index_type;

  bool is_in_range(const std::tuple<>& indices) const { return true; }
  bool is_in_range() const { return true; }

  index_t at(const std::tuple<>& indices) const { return 0; }
  index_t at() const { return 0; }
  index_t operator() (const std::tuple<>& indices) const { return 0; }
  index_t operator() () const { return 0; }

  index_t flat_extent() const { return 1; }
  index_t size() const { return 1; }
  bool empty() const { return false; }

  bool is_subset_of(const shape<>& other) const { return true; }
  bool is_one_to_one() const { return true; }
  bool is_dense() const { return true; }

  bool operator==(const shape<>& other) const { return true; }
  bool operator!=(const shape<>& other) const { return false; }
};

namespace internal {

// This genius trick is from
// https://github.com/halide/Halide/blob/30fb4fcb703f0ca4db6c1046e77c54d9b6d29a86/src/runtime/HalideBuffer.h#L2088-L2108
template<size_t D, typename Shape, typename Fn, typename... Indices,
  typename = decltype(std::declval<Fn>()(std::declval<Indices>()...))>
void for_all_indices_impl(int, const Shape& shape, Fn &&f, Indices... indices) {
  f(indices...);
}

template<size_t D, typename Shape, typename Fn, typename... Indices>
void for_all_indices_impl(double, const Shape& shape, Fn &&f, Indices... indices) {
  for (index_t i : shape.template dim<D>()) {
    for_all_indices_impl<D - 1>(0, shape, std::forward<Fn>(f), i, indices...);
  }
}

template<size_t D, typename Shape, typename Fn, typename... Indices,
  typename = decltype(std::declval<Fn>()(std::declval<std::tuple<Indices...>>()))>
void for_each_index_impl(int, const Shape& shape, Fn &&f, const std::tuple<Indices...>& indices) {
  f(indices);
}

template<size_t D, typename Shape, typename Fn, typename... Indices>
void for_each_index_impl(double, const Shape& shape, Fn &&f, const std::tuple<Indices...>& indices) {
  for (index_t i : shape.template dim<D>()) {
    for_each_index_impl<D - 1>(0, shape, std::forward<Fn>(f), std::tuple_cat(std::make_tuple(i), indices));
  }
}

}  // namespace internal

/** Iterate over all indices in the shape, calling a function fn for
 * each set of indices. The indices are in the same order as the dims
 * in the shape. The first dim is the 'inner' loop of the iteration,
 * and the last dim is the 'outer' loop. */
template <typename Shape, typename Fn>
void for_all_indices(const Shape& s, const Fn& fn) {
  internal::for_all_indices_impl<Shape::rank() - 1>(0, s, fn);
}
template <typename Shape, typename Fn>
void for_each_index(const Shape& s, const Fn& fn) {
  internal::for_each_index_impl<Shape::rank() - 1>(0, s, fn, std::tuple<>());
} 

/** Helper function to make a tuple from a variadic list of dims. */
template <typename... Dims>
auto make_shape(Dims... dims) {
  return shape<Dims...>(std::make_tuple(std::forward<Dims>(dims)...));
}

/** Helper function to make a dense shape from a variadic list of extents. */
template <typename... Extents>
auto make_dense_shape(index_t dim0_extent, Extents... extents) {
  return make_shape(dense_dim<>(dim0_extent), dim<>(extents)...);
}

namespace internal {

template <typename... Dims>
shape<Dims...> make_shape_from_tuple(const std::tuple<Dims...>& dims) {
  return shape<Dims...>(dims);
}

template <size_t Rank, size_t... Is>
auto make_default_dense_shape() {
  return make_shape_from_tuple(std::tuple_cat(std::make_tuple(dense_dim<>()),
                                              default_array_to_tuple<dim<>, Rank - 1>()));
}

template <typename Shape, size_t... Is>
auto make_dense_shape(const Shape& dims, std::index_sequence<Is...>) {
  return make_shape(dense_dim<>(std::get<0>(dims).min(), std::get<0>(dims).extent()),
                    dim<>(std::get<Is + 1>(dims).min(), std::get<Is + 1>(dims).extent())...);
}

}  // namespace internal

/** Create a new shape using a permutation DimIndices... of the
 * dimensions of the shape. */
template <size_t... DimIndices, typename Shape>
auto transpose(const Shape& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}

/** Make a shape with an equivalent domain of indices, but with dense
 * strides. */
template <typename... Dims>
auto make_dense_shape(const shape<Dims...>& shape) {
  constexpr int rank = sizeof...(Dims);
  return internal::make_dense_shape(shape.dims(), std::make_index_sequence<rank - 1>());
}

/** An arbitrary shape (no compile-time constant parameters) with the
 * specified Rank. */
template <size_t Rank>
using shape_of_rank =
  decltype(internal::make_shape_from_tuple(internal::default_array_to_tuple<dim<>, Rank>()));

/** An arbitrary dense shape with the specified Rank. Only the stride of
 * the innermost dimension is a compile-time constant one. */
template <size_t Rank>
using dense_shape = decltype(internal::make_default_dense_shape<Rank>());

/** A multi-dimensional wrapper around a pointer. */
template <typename T, typename Shape>
class array_ref {
  T* base_;
  Shape shape_;

 public:
  typedef T value_type;
  typedef Shape shape_type;
  typedef typename Shape::index_type index_type;
  typedef size_t size_type;
  typedef value_type& reference;
  typedef value_type* pointer;

  /** Make an array_ref to the given base pointer, interpreting it as
   * having the given shape. */
  array_ref(T* base, Shape shape) : base_(base), shape_(std::move(shape)) {}
  /** The copy constructor of a reference is a shallow copy. */
  array_ref(const array_ref& other) : array_ref(other.data(), other.shape()) {}

  /** Assigning an array_ref performs a copy or move assignment of each
   * element in this array from the corresponding element in 'other'. */
  array_ref& operator=(const array_ref& other) {
    if (this == &other) return *this;
    assign(other);
    return *this;
  }
  array_ref& operator=(array_ref&& other) {
    assign(other);
    return *this;
  }

  void assign(const array_ref& copy) const {
    if (this == &copy) return;
    if (!shape().is_shape_in_range(copy.shape())) {
      throw std::out_of_range("assignment accesses indices out of range of src");
    }
    for_each_index(shape(), [&](const index_type& x) {
      base_[shape_(x)] = copy(x);
    });
  }
  void assign(array_ref&& move) const {
    if (this == &move) return;
    if (!shape().is_shape_in_range(move.shape())) {
      throw std::out_of_range("assignment accesses indices out of range of src");
    }
    for_each_index(shape(), [&](const index_type& x) {
      base_[shape_(x)] = std::move(move(x));
    });
  }
  /** Copy-assign each element of this array to the given value. */
  void assign(const T& value) const {
    for_each_index(shape(), [&](const index_type& x) {
      base_[shape_(x)] = value;
    });
  }

  /** Allow conversion from array_ref<T> to array_ref<const T>. */
  operator array_ref<const T, Shape>() const {
    return array_ref<const T, Shape>(data(), shape());
  }

  /** Get a reference to the element at the given indices. If the
   * indices are out of range of the shape, throws
   * std::out_of_range. */
  template <typename... Indices>
  reference at(const std::tuple<Indices...>& indices) const {
    if (!shape_.is_in_range(indices)) {
      throw std::out_of_range("indices are out of range");
    }
    return base_[shape_.at(indices)];
  }
  template <typename... Indices>
  reference at(Indices... indices) const {
    return at(std::make_tuple(std::forward<Indices>(indices)...));
  }

  /** Get a reference to the element at the given indices. */
  template <typename... Indices>
  reference operator() (const std::tuple<Indices...>& indices) const {
    return base_[shape_(indices)];
  }
  template <typename... Indices>
  reference operator() (Indices... indices) const {
    return base_[shape_(std::forward<Indices>(indices)...)];
  }

  /** Call a function with a reference to each value in this array_ref. */
  template <typename Fn>
  void for_each_value(const Fn& fn) const {
    for_each_index(shape(), [&](const index_type& index) {
      fn(base_[shape_(index)]);
    });
  }

  /** Pointer to the start of the flattened array_ref. */
  pointer data() const { return base_; }

  /** Shape of this array_ref. */
  const Shape& shape() const { return shape_; }
  template <size_t D>
  const auto& dim() const { return shape().template dim<D>(); }
  /** Number of elements addressable by the shape of this array_ref. */
  size_type size() { return shape_.size(); }
  /** True if there are zero addressable elements by the shape of this
   * array_ref. */
  bool empty() const { return shape_.empty(); }
  /** True if this array_ref is dense in memory. */
  bool is_dense() const { return shape_.is_dense(); }

  /** Reshape the array_ref. The new shape must not address any elements
   * not already addressable by the current shape of this array_ref. */
  void reshape(Shape new_shape) {
    if (!new_shape.is_subset_of(shape())) {
      throw std::out_of_range("new_shape is not a subset of shape().");
    }
    shape_ = std::move(new_shape);
  }

  /** Compare the contents of this array_ref to the other array_ref. For two
   * array_refs to be considered equal, they must have the same shape, and
   * all elements addressable by the shape must also be equal. */
  bool operator!=(const array_ref& other) const {
    if (internal::mins(shape()) != internal::mins(other.shape()) ||
        internal::extents(shape()) != internal::extents(other.shape())) {
      return true;
    }

    // TODO: This currently calls operator!= on all elements of the
    // array_ref, even after we find a non-equal element.
    bool result = false;
    for_each_index(shape(), [&](const index_type& x) {
      if (base_[shape_(x)] != other(x)) {
        result = true;
      }
    });
    return result;
  }
  bool operator==(const array_ref& other) const {
    return !operator!=(other);
  }

  /** Reinterpret the data in this array_ref as a different type. */
  template <typename U>
  array_ref<U, Shape> reinterpret() {
    static_assert(sizeof(T) == sizeof(U), "sizeof(reinterpreted type U) != sizeof(array type T)");
    return array_ref<U, Shape>(reinterpret_cast<U*>(data()), shape());
  }
  template <typename U>
  array_ref<const U, Shape> reinterpret() const {
    static_assert(sizeof(T) == sizeof(U), "sizeof(reinterpreted type U) != sizeof(array type T)");
    return array_ref<const U, Shape>(reinterpret_cast<const U*>(data()), shape());
  }
};

/** array_ref with an arbitrary shape of the given Rank. */
template <typename T, size_t Rank>
using array_ref_of_rank = array_ref<T, shape_of_rank<Rank>>;

/** array_ref with a dense innermost dimension, and an arbitrary shape
 * otherwise, of the given Rank. */
template <typename T, size_t Rank>
using dense_array_ref = array_ref<T, dense_shape<Rank>>;

/** A multi-dimensional array container that mirrors std::vector. */
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
class array {
  Alloc alloc_;
  T* base_;
  Shape shape_;

  // After allocate the array is allocated but uninitialized.
  void allocate() {
    assert(!base_);
    index_t flat_extent = shape_.flat_extent();
    if (flat_extent > 0) {
      base_ = std::allocator_traits<Alloc>::allocate(alloc_, flat_extent);
    }
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
  void construct(const array& copy) {
    assert(base_ || shape_.empty());
    for_each_index(shape(), [&](const index_type& index) {
      std::allocator_traits<Alloc>::construct(alloc_, &operator()(index), copy(index));
    });
  }
  void construct(array&& move) {
    assert(base_ || shape_.empty());
    for_each_index(shape(), [&](const index_type& index) {
      std::allocator_traits<Alloc>::construct(alloc_, &operator()(index), std::move(move(index)));
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
      std::allocator_traits<Alloc>::deallocate(alloc_, base_, shape_.flat_extent());
      base_ = nullptr;
    }
  }

 public:
  typedef T value_type;
  typedef Shape shape_type;
  typedef Alloc allocator_type;
  typedef typename Shape::index_type index_type;
  typedef size_t size_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef typename std::allocator_traits<Alloc>::pointer pointer;
  typedef typename std::allocator_traits<Alloc>::const_pointer const_pointer;

  /** Construct an array with a default constructed Shape. Most shapes
   * by default are empty, but a Shape with non-zero compile-time
   * constants for all extents will be non-empty. */
  array() : array(Shape()) {}
  explicit array(const Alloc& alloc) : array(Shape(), alloc) {}
  /** Construct an array with a particular shape and value to
   * copy-construct all elements in the array. */
  array(Shape shape, const T& value, const Alloc& alloc = Alloc()) : array(alloc) {
    assign(std::move(shape), value);
  }
  /** Construct an array with a particular shape, with default
   * constructed elements. */
  explicit array(Shape shape, const Alloc& alloc = Alloc())
      : alloc_(alloc), base_(nullptr), shape_(std::move(shape)) {
    allocate();
    construct();
  }
  /** Copy construct from another array. This is a deep copy of the
   * contents of the array. */
  array(const array& copy)
      : array(std::allocator_traits<Alloc>::select_on_container_copy_construction(copy.get_allocator())) {
    assign(copy);
  }
  array(const array& copy, const Alloc& alloc) : array(alloc) {
    assign(copy);
  }
  /** Move construct from another array. If the allocator of this and
   * the other array are equal, this operation moves the allocation of
   * other to this array, and the other array becomes a default
   * constructed array. If the allocator of this and the other array
   * are non-equal, each element is move-constructed into a new
   * allocation. */
  array(array&& other) : array(std::move(other), Alloc()) {}
  array(array&& other, const Alloc& alloc) : array(alloc) {
    using std::swap;
    if (alloc_ != other.get_allocator()) {
      shape_ = other.shape_;
      allocate();
      construct(std::move(other));
    } else {
      swap(shape_, other.shape_);
      swap(base_, other.base_);
    }
  }
  ~array() { 
    deallocate(); 
  }

  array& operator=(const array& other) {
    if (this == &other) return *this;

    if (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
      deallocate();
      alloc_ = other.get_allocator();
    }

    assign(other);
    return *this;
  }
  array& operator=(array&& other) {
    if (std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value) {
      swap(other);
      other.clear();
    } else {
      assign(std::move(other));
    }
    return *this;
  }

  void assign(const array& copy) {
    if (this == &copy) return;
    if (shape_ == copy.shape()) {
      destroy();
    } else {
      deallocate();
      shape_ = copy.shape();
      allocate();
    }
    construct(copy);
  }
  void assign(array&& move) {
    if (this == &move) return;
    if (shape_ == move.shape()) {
      destroy();
    } else {
      deallocate();
      shape_ = move.shape();
      allocate();
    }
    construct(std::move(move));
  }
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

  /** Compute the flat offset of the indices. If an index is out of
   * range, throws std::out_of_range. */
  template <typename... Indices>
  reference at(const std::tuple<Indices...>& indices) {
    if (!shape_.is_in_range(indices)) {
      throw std::out_of_range("indices are out of range");
    }
    return base_[shape_.at(indices)];
  }
  template <typename... Indices>
  reference at(Indices... indices) {
    return at(std::make_tuple(std::forward<Indices>(indices)...));
  }
  template <typename... Indices>
  const_reference at(const std::tuple<Indices...>& indices) const {
    if (!shape_.is_in_range(indices)) {
      throw std::out_of_range("indices are out of range");
    }
    return base_[shape_.at(indices)];
  }
  template <typename... Indices>
  const_reference at(Indices... indices) const {
    return at(std::make_tuple(std::forward<Indices>(indices)...));
  }

  /** Compute the flat offset of the indices. */
  template <typename... Indices>
  reference operator() (const std::tuple<Indices...>& indices) {
    return base_[shape_(indices)];
  }
  template <typename... Indices>
  reference operator() (Indices... indices) {
    return base_[shape_(std::forward<Indices>(indices)...)];
  }
  template <typename... Indices>
  const_reference operator() (const std::tuple<Indices...>& indices) const {
    return base_[shape_(indices)];
  }
  template <typename... Indices>
  const_reference operator() (Indices... indices) const {
    return base_[shape_(std::forward<Indices>(indices)...)];
  }

  /** Call a function with a reference to each value in this array. */
  template <typename Fn>
  void for_each_value(const Fn& fn) {
    for_each_index(shape(), [&](const index_type& index) {
      fn(base_[shape_(index)]);
    });
  }
  template <typename Fn>
  void for_each_value(const Fn& fn) const {
    for_each_index(shape(), [&](const index_type& index) {
      fn(base_[shape_(index)]);
    });
  }

  /** Pointer to the start of the flattened array. */
  pointer data() { return base_; }
  const_pointer data() const { return base_; }

  /** Shape of this array. */
  const Shape& shape() const { return shape_; }
  template <size_t D>
  const auto& dim() const { return shape().template dim<D>(); }
  /** Number of elements addressable by the shape of this array. */
  size_type size() { return shape_.size(); }
  /** True if there are zero addressable elements by the shape of this
   * array. */
  bool empty() const { return shape_.empty(); }
  /** True if this array is dense in memory. */
  bool is_dense() const { return shape_.is_dense(); }
  /** Reset the shape of this array to default. */
  void clear() { 
    deallocate();
    shape_ = Shape();
    allocate();
  }

  /** Reshape the array. The new shape must not address any elements
   * not already addressable by the current shape of this array. */
  void reshape(Shape new_shape) {
    if (!new_shape.is_subset_of(shape())) {
      throw std::out_of_range("new_shape is not a subset of shape().");
    }
    shape_ = std::move(new_shape);
  }

  /** Compare the contents of this array to the other array. For two
   * arrays to be considered equal, they must have the same shape, and
   * all elements addressable by the shape must also be equal. */
  bool operator!=(const array& other) const {
    if (internal::mins(shape()) != internal::mins(other.shape()) ||
        internal::extents(shape()) != internal::extents(other.shape())) {
      return true;
    }

    // TODO: This currently calls operator!= on all elements of the
    // array, even after we find a non-equal element.
    bool result = false;
    for_each_index(shape(), [&](const index_type& x) {
      if (base_[shape_(x)] != other(x)) {
        result = true;
      }
    });
    return result;
  }
  bool operator==(const array& other) const {
    return !operator!=(other);
  }

  /** Swap the contents of two arrays. This performs zero copies or
   * moves of individual elements. */
  void swap(array& other) {
    using std::swap;

    // TODO: This probably should respect
    // std::allocator_traits<Alloc>::propagate_on_container_swap::value
    swap(alloc_, other.alloc_);
    swap(base_, other.base_);
    swap(shape_, other.shape_);
  }

  /** Make an array_ref referring to the data in this array. */
  array_ref<T, Shape> ref() {
    return array_ref<T, Shape>(data(), shape());
  }
  array_ref<const T, Shape> ref() const {
    return array_ref<const T, Shape>(data(), shape());
  }
  operator array_ref<T, Shape>() { return ref(); }
  operator array_ref<const T, Shape>() const { return ref(); }

  /** Reinterpret the data in this array as a different type. */
  template <typename U>
  array_ref<U, Shape> reinterpret() {
    static_assert(sizeof(T) == sizeof(U), "sizeof(reinterpreted type U) != sizeof(array type T)");
    return array_ref<U, Shape>(reinterpret_cast<U*>(data()), shape());
  }
  template <typename U>
  array_ref<const U, Shape> reinterpret() const {
    static_assert(sizeof(T) == sizeof(U), "sizeof(reinterpreted type U) != sizeof(array type T)");
    return array_ref<const U, Shape>(reinterpret_cast<const U*>(data()), shape());
  }
};

/** array with an arbitrary shape of the given Rank. */
template <typename T, size_t Rank, typename Alloc = std::allocator<T>>
using array_of_rank = array<T, shape_of_rank<Rank>, Alloc>;

/** array with a dense innermost dimension, and an arbitrary shape
 * otherwise. */
template <typename T, size_t Rank, typename Alloc = std::allocator<T>>
using dense_array = array<T, dense_shape<Rank>, Alloc>;

/** Make a new array from a shape. */
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

/** Copy an src array to a dest array. The range of the shape of dest
 * will be copied, and must be in range of src. */
template <typename T, typename ShapeSrc, typename ShapeDest>
void copy(const array_ref<T, ShapeSrc>& src,
          const array_ref<typename std::remove_const<T>::type, ShapeDest>& dest) {
  if (!dest.shape().is_shape_in_range(src.shape())) {
    throw std::out_of_range("dest indices are out of range of src");
  }
  for_each_index(dest.shape(), [&](const typename ShapeDest::index_type& index) {
    dest(index) = src(index);
  });
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocDest>
void copy(const array_ref<T, ShapeSrc>& src,
          array<typename std::remove_const<T>::type, ShapeDest, AllocDest>& dest) {
  copy(src, dest.ref());
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc>
void copy(const array<T, ShapeSrc, AllocSrc>& src, const array_ref<T, ShapeDest>& dest) {
  copy(src.ref(), dest);
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc, typename AllocDest>
void copy(const array<T, ShapeSrc, AllocSrc>& src, array<T, ShapeDest, AllocDest>& dest) {
  copy(src.ref(), dest.ref());
}

/** Make a copy of an array with a new shape. */
template <typename T, typename ShapeSrc, typename ShapeDest,
  typename AllocDest = std::allocator<T>>
auto make_copy(const array_ref<T, ShapeSrc>& src, const ShapeDest& copy_shape,
               const AllocDest& alloc = AllocDest()) {
  array<typename std::remove_const<T>::type, ShapeDest, AllocDest> dest(copy_shape, alloc);
  copy(src, dest);
  return dest;
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc,
  typename AllocDest = std::allocator<T>>
auto make_copy(const array<T, ShapeSrc, AllocSrc>& src, const ShapeDest& copy_shape,
               const AllocDest& alloc = AllocDest()) {
  return make_copy(src.ref(), copy_shape, alloc);
}

/** Make a copy of an array with the same shape as src, but with dense
 * strides. */
template <typename T, typename ShapeSrc, typename AllocDest = std::allocator<T>>
auto make_dense_copy(const array_ref<T, ShapeSrc>& src,
                     const AllocDest& alloc = AllocDest()) {
  return make_copy(src, make_dense_shape(src.shape()), alloc);
}
template <typename T, typename ShapeSrc, typename AllocSrc, typename AllocDest = std::allocator<T>>
auto make_dense_copy(const array<T, ShapeSrc, AllocSrc>& src,
                     const AllocDest& alloc = AllocDest()) {
  return make_dense_copy(src.ref(), alloc);
}

/** Allocator that owns a buffer of fixed size, which will be placed
 * on the stack if the owning container is allocated on the
 * stack. This can only be used with containers that have a maximum of
 * one live allocation, which is the case for array::array. */
// TODO: "stack_allocator" isn't a good name for this. It's a fixed
// allocation, but not necessarily a stack allocation.
template <class T, size_t N>
class stack_allocator {
  alignas(T) char alloc[N * sizeof(T)];
  bool allocated;

 public:  
  typedef T value_type;

  typedef std::false_type propagate_on_container_copy_assignment;
  typedef std::false_type propagate_on_container_move_assignment;
  typedef std::false_type propagate_on_container_swap;

  static stack_allocator select_on_container_copy_construction(const stack_allocator& a) {
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
    if (allocated) throw std::bad_alloc();
    if (n > N) throw std::bad_alloc();
    allocated = true;
    return reinterpret_cast<T*>(&alloc[0]);
  }
  void deallocate(T* p, size_t) noexcept {
    allocated = false;
  }

  template <class U, size_t U_N>
  friend bool operator==(const stack_allocator<T, N>& a, const stack_allocator<U, U_N>& b) {
    return &a.alloc[0] == &b.alloc[0];
  }

  template <class U, size_t U_N>
  friend bool operator!=(const stack_allocator<T, N>& a, const stack_allocator<U, U_N>& b) {
    return &a.alloc[0] != &b.alloc[0];
  }
};

}  // namespace array

#endif  // ARRAY_ARRAY_H

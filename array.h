#ifndef ARRAY_ARRAY_H
#define ARRAY_ARRAY_H

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>

#ifdef ARRAY_NO_EXCEPTIONS
#define ARRAY_THROW_OUT_OF_RANGE(m) do { assert(!m); abort(); } while(0)
#define ARRAY_THROW_BAD_ALLOC() do { assert(!"bad alloc"); abort(); } while(0)
#else
#define ARRAY_THROW_OUT_OF_RANGE(m) throw std::out_of_range(m)
#define ARRAY_THROW_BAD_ALLOC() throw std::bad_alloc();
#endif

namespace array {

typedef std::size_t size_t;
typedef std::ptrdiff_t index_t;

enum : index_t {
  /** This value indicates a compile-time constant stride is unknown,
   * and to use the corresponding runtime value instead. */
  // It would be better to use a more unreasonable value that would
  // never be used in practice. Fortunately, this does not affect
  // correctness, only performance, and it is hard to imagine a use
  // case for this where performance matters.
  UNK = -9,
};

#define ARRAY_CHECK_CONSTRAIT(constant, runtime) \
  assert(constant == runtime || constant == UNK);

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
 * enable providing compile time constants for the 'min', 'extent',
 * and 'stride' of the dim. These parameters define a mapping from the
 * indices of the dimension to offsets: offset(x) = (x - min)*stride.
 * The extent does not affect the mapping directly. Values not in the
 * range [min, min + extent) are considered to be out of bounds. */
template <index_t Min = UNK, index_t Extent = UNK, index_t Stride = UNK>
class dim {
 protected:
  index_t min_;
  index_t extent_;
  index_t stride_;

 public:
  /** Construct a new dim object. If the class template parameters
   * 'Min', 'Extent', or 'Stride' are not 'UNK', these runtime values
   * must match the compile-time values. */
  dim(index_t min, index_t extent, index_t stride = Stride)
    : min_(min), extent_(extent), stride_(stride) {
    ARRAY_CHECK_CONSTRAIT(Min, min);
    ARRAY_CHECK_CONSTRAIT(Extent, extent);
    ARRAY_CHECK_CONSTRAIT(Stride, stride);
  }
  dim(index_t extent = Extent) : dim(0, extent) {}
  dim(const dim&) = default;
  dim(dim&&) = default;
  /** Copy another dim object, possibly with different compile-time
   * template parameters. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride>
  dim(const dim<CopyMin, CopyExtent, CopyStride>& other)
      : dim(other.min(), other.extent(), other.stride()) {}

  dim& operator=(const dim&) = default;
  dim& operator=(dim&&) = default;
  /** Copy assignment of a dim object, possibly with different
   * compile-time template parameters. */
  template <index_t CopyMin, index_t CopyExtent, index_t CopyStride>
  dim& operator=(const dim<CopyMin, CopyExtent, CopyStride>& other) {
    ARRAY_CHECK_CONSTRAIT(Min, other.min());
    min_ = other.min();
    ARRAY_CHECK_CONSTRAIT(Extent, other.extent());
    extent_ = other.extent();
    ARRAY_CHECK_CONSTRAIT(Stride, other.stride());
    stride_ = other.stride();
    return *this;
  }

  /** Index of the first element in this dim. */
  index_t min() const { return internal::reconcile<Min>(min_); }
  void set_min(index_t min) {
    ARRAY_CHECK_CONSTRAIT(Min, min);
    min_ = min;
  }
  /** Number of elements in this dim. */
  index_t extent() const { return internal::reconcile<Extent>(extent_); }
  void set_extent(index_t extent) {
    ARRAY_CHECK_CONSTRAIT(Extent, extent);
    extent_ = extent;
  }
  /** Distance in flat indices between neighboring elements in this dim. */
  index_t stride() const { return internal::reconcile<Stride>(stride_); }
  void set_stride(index_t stride) {
    ARRAY_CHECK_CONSTRAIT(Stride, stride);
    stride_ = stride;
  }
  /** Index of the last element in this dim. */
  index_t max() const { return min() + extent() - 1; }

  /** Offset of the index 'at' in this dim. */
  index_t offset(index_t at) const { return (at - min()) * stride(); }

  /** Returns true if 'at' is within the range [min(), max()]. */
  bool is_in_range(index_t at) const { return min() <= at && at <= max(); }

  /** Make an iterator referring to the first element in this dim. */
  dim_iterator begin() const { return dim_iterator(min()); }
  /** Make an iterator referring to one past the last element in this
   * dim. */
  dim_iterator end() const { return dim_iterator(max() + 1); }

  /** Two dim objects are considered equal if their mins, extents, and
   * strides are equal. */
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride>
  bool operator==(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
    return min() == other.min() && extent() == other.extent() && stride() == other.stride();
  }
  template <index_t OtherMin, index_t OtherExtent, index_t OtherStride>
  bool operator!=(const dim<OtherMin, OtherExtent, OtherStride>& other) const {
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

/** A specialization of 'dim' where the compile-time stride parameter
 * is known to be one. */
template <index_t Min = UNK, index_t Extent = UNK>
using dense_dim = dim<Min, Extent, 1>;

/** A specialization of 'dim' where only the stride parameter is
 * specified at compile time. */
template <index_t Stride>
using strided_dim = dim<UNK, UNK, Stride>;

/** A specialization of 'dim' where the compile-time stride parameter
 * is known to be zero. */
template <index_t Min = UNK, index_t Extent = UNK>
using broadcast_dim = dim<Min, Extent, 0>;

/** Clamp an index to the range [min, max]. */
inline index_t clamp(index_t x, index_t min, index_t max) {
  return std::min(std::max(x, min), max);
}

/** Clamp an index to the range described by a dim. */
template <typename Dim>
inline index_t clamp(index_t x, const Dim& d) {
  return clamp(x, d.min(), d.max());
}

namespace internal {

// Some variadic reduction helpers.
inline index_t sum() { return 0; }
inline index_t product() { return 1; }
inline index_t variadic_max() { return std::numeric_limits<index_t>::min(); }

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

template <typename... Dims, size_t... Is>
std::array<dim<>, sizeof...(Is)> dims_as_array_impl(const std::tuple<Dims...>& dims, std::index_sequence<Is...>) {
  return {{std::get<Is>(dims)...}};
}

template <typename... Dims>
std::array<dim<>, sizeof...(Dims)> dims_as_array(const std::tuple<Dims...>& dims) {
  return dims_as_array_impl(dims, std::make_index_sequence<sizeof...(Dims)>());
}

// https://github.com/halide/Halide/blob/fc8cfb078bed19389f72883a65d56d979d18aebe/src/runtime/HalideBuffer.h#L43-L63
// A helper to check if a parameter pack is entirely implicitly
// int-convertible to use with std::enable_if
template<typename... Args>
struct all_integral : std::false_type {};

template<>
struct all_integral<> : std::true_type {};

template<typename T, typename... Args>
struct all_integral<T, Args...> {
  static constexpr bool value =
      std::is_convertible<T, index_t>::value && all_integral<Args...>::value;
};

// Floats and doubles are technically implicitly int-convertible, but
// doing so produces a warning we treat as an error, so just disallow
// it here.
template<typename... Args>
struct all_integral<float, Args...> : std::false_type {};

template<typename... Args>
struct all_integral<double, Args...> : std::false_type {};

}  // namespace internal

/** A list of 'dim' objects describing a multi-dimensional space of
 * indices. The 'rank' of a shape refers to the number of dimensions
 * in the shape. Shapes map multiple dim objects to offsets by adding
 * each mapping dim to offset together to produce a 'flat offset'. The
 * first dimension is known as the 'innermost' dimension, and
 * dimensions then increase until the 'outermost' dimension. */
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
  /** Construct this shape from a different type of
   * shape. 'conversion' must be convertible to this shape. */
  template <typename... OtherDims>
  shape(const shape<OtherDims...>& conversion)
    : dims_(internal::convert_tuple<Dims...>(conversion.dims())) {}

  shape& operator=(const shape&) = default;
  shape& operator=(shape&&) = default;

  /** Assign this shape from a different type of shape. 'conversion'
   * must be convertible to this shape. */
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

  /** Returns true if the index 'indices' are in range of this shape. */
  bool is_in_range(const index_type& indices) const {
    return internal::is_in_range(dims_, indices);
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  bool is_in_range(Indices... indices) const {
    return is_in_range(std::make_tuple(std::forward<Indices>(indices)...));
  }

  /** Returns true if all of the indices in bounds of 'other_shape'
   * are in bounds of this shape. */
  template <typename... OtherDims>
  bool is_shape_in_range(const shape<OtherDims...>& other_shape) const {
    return is_in_range(internal::mins(other_shape)) && is_in_range(internal::maxes(other_shape));
  }

  /** Compute the flat offset of the index 'indices'. If the
   * 'indices' are out of range of this shape, throws
   * std::out_of_range. */
  index_t at(const index_type& indices) const {
    if (!is_in_range(indices)) {
      ARRAY_THROW_OUT_OF_RANGE("indices are out of range");
    }
    return internal::flat_offset(dims_, indices);
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  index_t at(Indices... indices) const {
    return at(std::make_tuple(std::forward<Indices>(indices)...));
  }
  /** Compute the flat offset of the index 'indices'. */
  index_t operator() (const index_type& indices) const {
    return internal::flat_offset(dims_, indices);
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  index_t operator() (Indices... indices) const {
    return (*this)(std::make_tuple(std::forward<Indices>(indices)...));
  }

  /** Get a specific dim of this shape. */
  template <size_t D>
  auto& dim() { return std::get<D>(dims_); }
  template <size_t D>
  const auto& dim() const { return std::get<D>(dims_); }
  /** Get a specific dim of this shape with a runtime dimension
   * d. This will lose knowledge of any compile-time constant
   * dimension attributes. */
  array::dim<> dim(size_t d) const { return internal::dims_as_array(dims_)[d]; }

  /** Get a tuple of all of the dims of this shape. */
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
   * zero, multiple indices will map to the same flat index; in this
   * case, this function will return false. */
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

  /** Returns true if this shape is 'compact' in memory. A shape is
   * compact if there are no unaddressable flat indices between the
   * first and last addressable flat elements. */
  bool is_compact() const { return size() == flat_extent(); }

  /** Provide some aliases for common interpretations of
   * dimensions. */
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

  /** Assuming this array represents an image with dimensions {width,
   * height, channels}, get the extent of those dimensions. */
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

// TODO: Try to avoid needing this specialization. The only reason
// it is necessary is because the above defines two default
// constructors in the case of a scalar shape.
template <>
class shape<> {
 public:
  shape() {}

  static constexpr size_t rank() { return 0; }
  static constexpr bool is_scalar() { return true; }

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
  bool is_compact() const { return true; }

  bool operator==(const shape<>& other) const { return true; }
  bool operator!=(const shape<>& other) const { return false; }
};

namespace internal {

template <index_t Min, index_t Extent, index_t Stride, typename DimSrc>
bool is_dim_compatible(const dim<Min, Extent, Stride>& dest, const DimSrc& src) {
  return
    (Min == UNK || src.min() == Min) &&
    (Extent == UNK || src.extent() == Extent) &&
    (Stride == UNK || src.stride() == Stride);
}

template <typename... DimsDest, typename ShapeSrc, size_t... Is>
bool is_shape_compatible(const shape<DimsDest...>& dest, const ShapeSrc& src, std::index_sequence<Is...>) {
  return sum((is_dim_compatible(DimsDest(), src.template dim<Is>()) ? 0 : 1)...) == 0;
}

}  // namespace internal

/** Test if a shape 'src' can be assigned to a shape of type
 * 'ShapeDest' without error. */
template <typename ShapeDest, typename ShapeSrc>
bool is_compatible(const ShapeSrc& src) {
  static_assert(ShapeSrc::rank() == ShapeDest::rank(), "shapes must have the same rank.");
  return internal::is_shape_compatible(ShapeDest(), src, std::make_index_sequence<ShapeSrc::rank()>());
}

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

/** Iterate over all indices in the shape, calling a function 'fn' for
 * each set of indices. The indices are in the same order as the dims
 * in the shape. The first dim is the 'inner' loop of the iteration,
 * and the last dim is the 'outer' loop. 'for_all_indices' calls 'fn'
 * with a list of arguments corresponding to each dim. 'for_each_index'
 * calls 'fn' with a Shape::index_type object describing the indices. */
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
  return shape<Dims...>(std::forward<Dims>(dims)...);
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

template <typename Shape, size_t N>
Shape make_shape_from_array(const std::array<dim<>, N>& dims) {
  return Shape(dims);
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

template <typename... Dims, size_t... Is>
shape<Dims...> make_compact_shape(const shape<Dims...>& s, std::index_sequence<Is...>) {
  return {{s.template dim<Is>().min(), s.template dim<Is>().extent()}...};
}

}  // namespace internal

/** Create a new shape using a permutation DimIndices... of the
 * dimensions of the shape. */
template <size_t... DimIndices, typename Shape>
auto permute(const Shape& shape) {
  return make_shape(shape.template dim<DimIndices>()...);
}

/** Make a shape with an equivalent domain of indices, but with dense
 * strides. */
template <typename... Dims>
auto make_dense(const shape<Dims...>& shape) {
  constexpr int rank = sizeof...(Dims);
  return internal::make_dense_shape(shape.dims(), std::make_index_sequence<rank - 1>());
}

/** Make a shape with an equivalent domain of indices, but with compact
 * strides. Only required strides are respected. */
template <typename Shape>
Shape make_compact(const Shape& s) {
  return internal::make_compact_shape(s, std::make_index_sequence<Shape::rank()>());
}

/** An arbitrary shape (no compile-time constant parameters) with the
 * specified Rank. */
template <size_t Rank>
using shape_of_rank =
  decltype(internal::make_shape_from_tuple(internal::default_array_to_tuple<dim<>, Rank>()));

/** A shape where the innermost dimension is a 'dense_dim' object, and
 * all other dimensions are arbitrary. */
template <size_t Rank>
using dense_shape = decltype(internal::make_default_dense_shape<Rank>());

namespace internal {

// Sort the dims such that strides are increasing from dim 0, and
// contiguous dimensions are fused.
template <typename Shape>
shape_of_rank<Shape::rank()> optimize_shape(const Shape& shape) {
  auto dims = internal::dims_as_array(shape.dims());

  // Sort the dims by stride.
  std::sort(dims.begin(), dims.end(), [](const dim<>& l, const dim<>& r) {
    return l.stride() < r.stride();
  });

  // Find dimensions that are contiguous and fuse them.
  size_t rank = dims.size();
  for (size_t i = 0; i + 1 < rank;) {
    if (dims[i].stride() * dims[i].extent() == dims[i + 1].stride()) {
      // These two dimensions are contiguous. Fuse them and move
      // the rest of the dimensions up to replace the fused dimension.
      dims[i].set_min(dims[i].min() + dims[i + 1].min() * dims[i + 1].stride());
      dims[i].set_extent(dims[i].extent() * dims[i + 1].extent());
      for (size_t j = i + 1; j + 1 < rank; j++) {
	dims[j] = dims[j + 1];
      }
      rank--;
    } else {
      i++;
    }
  }

  // Unfortunately, we can't make the rank of the resulting shape smaller.
  // Fill the end of the array with size 1 dimensions.
  for (size_t i = rank; i < dims.size(); i++) {
    dims[i] = dim<>(0, 1, dims[i - 1].stride() * dims[i - 1].extent());
  }

  return make_shape_from_array<shape_of_rank<Shape::rank()>>(dims);
}

// Call fn on the values addressed by shape in any order.
template <typename T, typename Shape, typename Fn>
void for_each_value(T* base, const Shape& shape, const Fn& fn) {
  auto opt_shape = optimize_shape(shape);

  // If the optimized shape's first dimension is 1, we can convert
  // this to a dense shape. This may help the compiler optimize this
  // further.
  typedef typename Shape::index_type index_type;
  if (opt_shape.template dim<0>().stride() == 1) {
    dense_shape<Shape::rank()> dense_opt_shape = opt_shape;
    for_each_index(dense_opt_shape, [&](const index_type& index) {
      fn(base[dense_opt_shape(index)]);
    });
  } else {
    for_each_index(opt_shape, [&](const index_type& index) {
      fn(base[opt_shape(index)]);
    });
  }
}

// Call fn on the values of src and dest both addressed by shape in
// any order.
template <typename TSrc, typename TDest, typename Shape, typename Fn>
void for_each_src_dest(TSrc* src, TDest* dest, const Shape& shape, const Fn& fn) {
  auto opt_shape = optimize_shape(shape);

  // If the optimized shape's first dimension is 1, we can convert
  // this to a dense shape. This may help the compiler optimize this
  // further.
  typedef typename Shape::index_type index_type;
  if (opt_shape.template dim<0>().stride() == 1) {
    dense_shape<Shape::rank()> dense_opt_shape = opt_shape;
    for_each_index(dense_opt_shape, [&](const index_type& index) {
      index_t flat_index = dense_opt_shape(index);
      fn(src[flat_index], dest[flat_index]);
    });
  } else {
    for_each_index(opt_shape, [&](const index_type& index) {
      index_t flat_index = opt_shape(index);
      fn(src[flat_index], dest[flat_index]);
    });
  }
}

// Optimize a src and dest shape. The dest shape is made dense, and contiguous
// dimensions are fused.
template <typename ShapeSrc, typename ShapeDest>
std::pair<shape_of_rank<ShapeSrc::rank()>, shape_of_rank<ShapeDest::rank()>>
optimize_src_dest_shapes(const ShapeSrc& src, const ShapeDest& dest) {
  constexpr size_t rank = ShapeSrc::rank();
  static_assert(rank == ShapeDest::rank(), "copy shapes must have same rank.");
  auto src_dims = internal::dims_as_array(src.dims());
  auto dest_dims = internal::dims_as_array(dest.dims());

  struct copy_dims {
    dim<> src;
    dim<> dest;
  };
  std::array<copy_dims, rank> dims;
  for (int i = 0; i < rank; i++) {
    dims[i] = {src_dims[i], dest_dims[i]};
  }

  // Sort the dims by the dest stride.
  std::sort(dims.begin(), dims.end(), [](const copy_dims& l, const copy_dims& r) {
    return l.dest.stride() < r.dest.stride();
  });

  // Find dimensions that are contiguous and fuse them.
  size_t new_rank = dims.size();
  for (size_t i = 0; i + 1 < new_rank;) {
    if (dims[i + 1].src.stride() == dims[i + 1].dest.stride() &&
	dims[i].src.stride() * dims[i].src.extent() == dims[i + 1].src.stride() &&
	dims[i].dest.stride() * dims[i].dest.extent() == dims[i + 1].dest.stride()) {
      // These two dimensions are contiguous. Fuse them and move
      // the rest of the dimensions up to replace the fused dimension.
      dims[i].src.set_min(dims[i].src.min() + dims[i + 1].src.min() * dims[i + 1].src.stride());
      dims[i].src.set_extent(dims[i].src.extent() * dims[i + 1].src.extent());
      dims[i].dest.set_min(dims[i].dest.min() + dims[i + 1].dest.min() * dims[i + 1].dest.stride());
      dims[i].dest.set_extent(dims[i].dest.extent() * dims[i + 1].dest.extent());
      for (size_t j = i + 1; j + 1 < new_rank; j++) {
	dims[j] = dims[j + 1];
      }
      new_rank--;
    } else {
      i++;
    }
  }

  // Unfortunately, we can't make the rank of the resulting shape dynamic.
  // Fill the end of the array with size 1 dimensions.
  for (size_t i = new_rank; i < dims.size(); i++) {
    dims[i] = {
      dim<>(0, 1, dims[i - 1].src.stride() * dims[i - 1].src.extent()),
      dim<>(0, 1, dims[i - 1].dest.stride() * dims[i - 1].dest.extent()),
    };
  }

  for (int i = 0; i < dims.size(); i++) {
    src_dims[i] = dims[i].src;
    dest_dims[i] = dims[i].dest;
  }

  return {
      make_shape_from_array<shape_of_rank<rank>>(src_dims),
      make_shape_from_array<shape_of_rank<rank>>(dest_dims),
  };
}

// Call fn on the values of src and dest in any order.
template <typename TSrc, typename TDest, typename ShapeSrc, typename ShapeDest, typename Fn>
void for_each_src_dest(TSrc* src, TDest* dest, const ShapeSrc& shape_src, const ShapeDest& shape_dest,
		       const Fn& fn) {
  if (shape_dest.empty()) {
    return;
  }
  if (!shape_src.is_shape_in_range(shape_dest)) {
    ARRAY_THROW_OUT_OF_RANGE("dest indices out of range of src");
  }

  // For this function, we don't care about the order in which the
  // callback is called. Optimize the shapes for memory access order.
  auto opt_shape = optimize_src_dest_shapes(shape_src, shape_dest);
  const auto& opt_shape_src = opt_shape.first;
  const auto& opt_shape_dest = opt_shape.second;

  // If the optimized shape's first dimension is 1, we can convert
  // this to a dense shape. This may help the compiler optimize this
  // further.
  typedef typename ShapeDest::index_type index_type;
  if (opt_shape_src.template dim<0>().stride() == 1 &&
      opt_shape_dest.template dim<0>().stride() == 1) {
    dense_shape<ShapeSrc::rank()> dense_opt_shape_src = opt_shape_src;
    dense_shape<ShapeDest::rank()> dense_opt_shape_dest = opt_shape_dest;
    for_each_index(dense_opt_shape_dest, [&](const index_type& index) {
      fn(src[dense_opt_shape_src(index)], dest[dense_opt_shape_dest(index)]);
    });
  } else if (opt_shape_dest.template dim<0>().stride() == 1) {
    dense_shape<ShapeDest::rank()> dense_opt_shape_dest = opt_shape_dest;
    for_each_index(dense_opt_shape_dest, [&](const index_type& index) {
      fn(src[opt_shape_src(index)], dest[dense_opt_shape_dest(index)]);
    });
  } else {
    for_each_index(opt_shape_dest, [&](const index_type& index) {
      fn(src[opt_shape_src(index)], dest[opt_shape_dest(index)]);
    });
  }
  // The dense src, non-dense dest case is omitted. That seems unlikely
  // given that we sorted the dimensions to make the dest dense,
  // but it could happen if the dest does not have any dense dimensions,
  // and the source has one in the same place as the innermost dimension
  // of dest.
}

}  // namespace internal

/** A reference to an array is an object with a shape mapping indices
 * to flat offsets, which are used to dereference a pointer. This object
 * has 'reference semantics':
 * - O(1) copy construction, cheap to pass by value.
 * - Cannot be reassigned. */
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

  /** Make an array_ref to the given 'base' pointer, interpreting it as
   * having the shape 'shape'. */
  array_ref(T* base, Shape shape) : base_(base), shape_(std::move(shape)) {}
  /** The copy constructor of a ref is a shallow copy. */
  array_ref(const array_ref& other) = default;

  /** Assigning an array_ref performs a copy or move assignment of each
   * element in this array from the corresponding element in 'other'. */
  array_ref& operator=(const array_ref& other) {
    assign(other);
    return *this;
  }
  array_ref& operator=(array_ref&& other) {
    assign(other);
    return *this;
  }

  void assign(const array_ref& other) const {
    if (data() == other.data()) {
      assert(shape() == other.shape());
      return;
    }
    internal::for_each_src_dest(other.data(), data(), other.shape(), shape(),
				[&](value_type& src, value_type& dest) {
      dest = src;
    });
  }
  void assign(array_ref&& other) const {
    if (data() == other.data()) {
      assert(shape() == other.shape());
      return;
    }
    internal::for_each_src_dest(other.data(), data(), other.shape(), shape(),
				[&](value_type& src, value_type& dest) {
      dest = std::move(src);
    });
  }
  /** Copy-assign each element of this array to the given value. */
  void assign(const T& value) const {
    internal::for_each_value(data(), shape(), [&](T& x) { x = value; });
  }

  /** Get a reference to the element at the given 'indices'. If the
   * 'indices' are out of range of 'shape()', throws
   * std::out_of_range. */
  reference at(const index_type& indices) const {
    return base_[shape_.at(indices)];
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  reference at(Indices... indices) const {
    return base_[shape_.at(std::forward<Indices>(indices)...)];
  }

  /** Get a reference to the element at the given indices. */
  reference operator() (const index_type& indices) const {
    return base_[shape_(indices)];
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  reference operator() (Indices... indices) const {
    return base_[shape_(std::forward<Indices>(indices)...)];
  }

  /** Call a function with a reference to each value in this
   * array_ref. The order in which 'fn' is called is undefined. */
  template <typename Fn>
  void for_each_value(const Fn& fn) const {
    internal::for_each_value(data(), shape(), fn);
  }

  /** Pointer to the start of the flattened array_ref. */
  pointer data() const { return base_; }

  /** Shape of this array_ref. */
  const Shape& shape() const { return shape_; }

  static constexpr size_t rank() { return Shape::rank(); }
  static constexpr bool is_scalar() { return Shape::is_scalar(); }
  template <size_t D>
  const auto& dim() const { return shape().template dim<D>(); }
  array::dim<> dim(size_t d) const { return shape().dim(d); }
  size_type size() const { return shape_.size(); }
  bool empty() const { return shape_.empty(); }
  bool is_compact() const { return shape_.is_compact(); }

  /** Provide some aliases for common interpretations of
   * dimensions. */
  const auto& i() const { return shape().i(); }
  const auto& j() const { return shape().j(); }
  const auto& k() const { return shape().k(); }

  const auto& x() const { return shape().x(); }
  const auto& y() const { return shape().y(); }
  const auto& z() const { return shape().z(); }
  const auto& w() const { return shape().w(); }

  const auto& c() const { return shape().c(); }

  /** Assuming this array represents an image with dimensions width,
   * height, channels, get the extent of those dimensions. */
  index_t width() const { return shape().width(); }
  index_t height() const { return shape().height(); }
  index_t channels() const { return shape().channels(); }

  /** Assuming this array represents a matrix with dimensions {rows,
   * cols}, get the extent of those dimensions. */
  index_t rows() const { return shape().rows(); }
  index_t columns() const { return shape().columns(); }

  /** Compare the contents of this array_ref to 'other'. For two
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
    internal::for_each_src_dest(other.data(), data(), other.shape(), shape(),
				[&](const value_type& src, const value_type& dest) {
      if (src != dest) {
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

  /** Allow conversion from array_ref<T> to array_ref<const T>. */
  operator array_ref<const T, Shape>() const {
    return array_ref<const T, Shape>(data(), shape());
  }

  /** Change the shape of the array to 'new_shape', and move the base
   * pointer by 'offset'. */
  void reshape(Shape new_shape, index_t offset = 0) {
    shape_ = std::move(new_shape);
    base_ += offset;
  }
};

/** array_ref with an arbitrary shape of the compile-time constant
 * 'Rank'. */
template <typename T, size_t Rank>
using array_ref_of_rank = array_ref<T, shape_of_rank<Rank>>;

/** array_ref with a 'dense_dim' innermost dimension, and an arbitrary
 * shape otherwise, of the compile-time constant 'Rank'. */
template <typename T, size_t Rank>
using dense_array_ref = array_ref<T, dense_shape<Rank>>;

/** A multi-dimensional array container that owns an allocation of
 * memory. This container is designed to mirror the semantics of
 * std::vector where possible. */
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
class array {
  Alloc alloc_;
  T* buffer_;
  index_t buffer_size_;
  T* base_;
  Shape shape_;

  // After allocate the array is allocated but uninitialized.
  void allocate() {
    assert(!buffer_);
    index_t flat_extent = shape_.flat_extent();
    if (flat_extent > 0) {
      buffer_size_ = flat_extent;
      buffer_ = std::allocator_traits<Alloc>::allocate(alloc_, flat_extent);
    }
    base_ = buffer_;
  }

  // Call the constructor on all of the elements of the array.
  void construct() {
    assert(base_ || shape_.empty());
    internal::for_each_value(base_, shape_, [&](T& x) {
      std::allocator_traits<Alloc>::construct(alloc_, &x);
    });
  }
  void construct(const T& init) {
    assert(base_ || shape_.empty());
    internal::for_each_value(base_, shape_, [&](T& x) {
      std::allocator_traits<Alloc>::construct(alloc_, &x, init);
    });
  }
  void copy_construct(const array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape());
    internal::for_each_src_dest(other.data(), base_, shape_,
				[&](const value_type& src, value_type& dest) {
      std::allocator_traits<Alloc>::construct(alloc_, &dest, src);
    });
  }
  void move_construct(array& other) {
    assert(base_ || shape_.empty());
    assert(shape_ == other.shape());
    internal::for_each_src_dest(other.data(), base_, shape_,
				[&](value_type& src, value_type& dest) {
      std::allocator_traits<Alloc>::construct(alloc_, &dest, std::move(src));
    });
  }

  // Call the destructor on every element.
  void destroy() {
    assert(base_ || shape_.empty());
    internal::for_each_value(base_, shape_, [&](T& x) {
      std::allocator_traits<Alloc>::destroy(alloc_, &x);
    });
  }

  void deallocate() {
    if (base_) {
      destroy();
      base_ = nullptr;
      std::allocator_traits<Alloc>::deallocate(alloc_, buffer_, buffer_size_);
      buffer_ = nullptr;
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
  /** Construct an array with a particular 'shape', allocated by
   * 'alloc'. All elements in the array are copy-constructed from
   * 'value'. */
  array(Shape shape, const T& value, const Alloc& alloc = Alloc()) : array(alloc) {
    assign(std::move(shape), value);
  }
  /** Construct an array with a particular 'shape', allocated by
   * 'alloc', with default constructed elements. */
  explicit array(Shape shape, const Alloc& alloc = Alloc())
      : alloc_(alloc), buffer_(nullptr), buffer_size_(0), base_(nullptr), shape_(std::move(shape)) {
    allocate();
    construct();
  }
  /** Copy construct from another array 'other', using copy's
   * allocator. This is a deep copy of the contents of 'other'. */
  array(const array& other)
      : array(std::allocator_traits<Alloc>::select_on_container_copy_construction(other.get_allocator())) {
    assign(other);
  }
  /** Copy construct from another array 'other'. The array is allocated
   * using 'alloc'. This is a deep copy of the contents of 'other'. */
  array(const array& other, const Alloc& alloc) : array(alloc) {
    assign(other);
  }
  /** Move construct from another array 'other'. If the allocator of
   * this array and the other array are equal, this operation moves
   * the allocation of other to this array, and the other array
   * becomes a default constructed array. If the allocator of this and
   * the other array are non-equal, each element is move-constructed
   * into a new allocation. */
  array(array&& other) : array(std::move(other), Alloc()) {}
  array(array&& other, const Alloc& alloc) : array(alloc) {
    using std::swap;
    if (alloc_ != other.get_allocator()) {
      shape_ = other.shape_;
      allocate();
      move_construct(other);
    } else {
      swap(buffer_, other.buffer_);
      swap(buffer_size_, other.buffer_size_);
      swap(base_, other.base_);
      swap(shape_, other.shape_);
    }
  }
  ~array() {
    deallocate();
  }

  /** Assign the contents of the array as a copy of 'other'. The array
   * is deallocated if the allocator cannot be propagated on
   * assignment. The array is then reallocated if necessary, and each
   * element in the array is copy constructed from other. */
  array& operator=(const array& other) {
    if (data() == other.data()) {
      assert(shape() == other.shape());
      return *this;
    }

    if (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
      deallocate();
      alloc_ = other.get_allocator();
    }

    assign(other);
    return *this;
  }
  /** Assign the contents of the array by moving from 'other'. If the
   * allocator can be propagated on move assignment, the allocation of
   * 'other' is moved in an O(1) operation. If the allocator cannot be
   * propagated, each element is move-assigned from 'other'. */
  array& operator=(array&& other) {
    if (data() == other.data()) {
      assert(shape() == other.shape());
      return *this;
    }

    if (std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value) {
      swap(other);
      other.clear();
    } else {
      assign(std::move(other));
    }
    return *this;
  }

  /** Assign the contents of the array to be a copy or move of 'other'. The
   * array is destroyed, reallocated if necessary, and then each
   * element is copy- or move-constructed from 'other'. */
  void assign(const array& other) {
    if (data() == other.data()) {
      assert(shape() == other.shape());
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
    if (data() == other.data()) {
      assert(shape() == other.shape());
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

  /** Assign the contents of this array to have 'shape' with each
   * element copy constructed from 'value'. */
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
   * bounds, throws std::out_of_range. */
  reference at(const index_type& indices) {
    return base_[shape_.at(indices)];
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  reference at(Indices... indices) {
    return base_[shape_.at(std::forward<Indices>(indices)...)];
  }
  const_reference at(const index_type& indices) const {
    return base_[shape_.at(indices)];
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  const_reference at(Indices... indices) const {
    return base_[shape_.at(std::forward<Indices>(indices)...)];
  }

  /** Compute the flat offset of the indices. Does not check if the
   * indices are in bounds. */
  reference operator() (const index_type& indices) {
    return base_[shape_(indices)];
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  reference operator() (Indices... indices) {
    return base_[shape_(std::forward<Indices>(indices)...)];
  }
  const_reference operator() (const index_type& indices) const {
    return base_[shape_(indices)];
  }
  template <typename... Indices,
      typename = typename std::enable_if<internal::all_integral<Indices...>::value>::type>
  const_reference operator() (Indices... indices) const {
    return base_[shape_(std::forward<Indices>(indices)...)];
  }

  /** Call a function with a reference to each value in this
   * array. The order in which 'fn' is called is undefined. */
  template <typename Fn>
  void for_each_value(const Fn& fn) {
    internal::for_each_value(data(), shape(), fn);
  }
  template <typename Fn>
  void for_each_value(const Fn& fn) const {
    internal::for_each_value(data(), shape(), fn);
  }

  /** Pointer to the start of the flat array, which is a pointer to
   * the min index of the shape. */
  pointer data() { return base_; }
  const_pointer data() const { return base_; }

  /** Shape of this array. */
  const Shape& shape() const { return shape_; }

  static constexpr size_t rank() { return Shape::rank(); }
  static constexpr bool is_scalar() { return Shape::is_scalar(); }
  template <size_t D>
  const auto& dim() const { return shape().template dim<D>(); }
  ::array::dim<> dim(size_t d) const { return shape().dim(d); }
  size_type size() const { return shape_.size(); }
  bool empty() const { return shape_.empty(); }
  bool is_compact() const { return shape_.is_compact(); }

  /** Reset the shape of this array to default. */
  void clear() {
    deallocate();
    shape_ = Shape();
    allocate();
  }

  /** Provide some aliases for common interpretations of
   * dimensions. */
  const auto& i() const { return shape().i(); }
  const auto& j() const { return shape().j(); }
  const auto& k() const { return shape().k(); }

  const auto& x() const { return shape().x(); }
  const auto& y() const { return shape().y(); }
  const auto& z() const { return shape().z(); }
  const auto& w() const { return shape().w(); }

  const auto& c() const { return shape().c(); }

  /** Assuming this array represents an image with dimensions width,
   * height, channels, get the extent of those dimensions. */
  index_t width() const { return shape().width(); }
  index_t height() const { return shape().height(); }
  index_t channels() const { return shape().channels(); }

  /** Assuming this array represents a matrix with dimensions {rows,
   * cols}, get the extent of those dimensions. */
  index_t rows() const { return shape().rows(); }
  index_t columns() const { return shape().columns(); }

  /** Compare the contents of this array to 'other'. For two arrays to
   * be considered equal, they must have the same shape, and all
   * elements addressable by the shape must also be equal. */
  bool operator!=(const array& other) const {
    return ref() != other.ref();
  }
  bool operator==(const array& other) const {
    return ref() == other.ref();
  }

  /** Swap the contents of two arrays. This performs zero copies or
   * moves of individual elements. */
  void swap(array& other) {
    using std::swap;

    // TODO: This probably should respect
    // std::allocator_traits<Alloc>::propagate_on_container_swap::value
    swap(alloc_, other.alloc_);
    swap(buffer_, other.buffer_);
    swap(buffer_size_, other.buffer_size_);
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

  /** Change the shape of the array to 'new_shape', and move the base
   * pointer by 'offset'. */
  void reshape(Shape new_shape, index_t offset = 0) {
    shape_ = std::move(new_shape);
    base_ += offset;
  }
};

/** An array type with an arbitrary shape of rank 'Rank'. */
template <typename T, size_t Rank, typename Alloc = std::allocator<T>>
using array_of_rank = array<T, shape_of_rank<Rank>, Alloc>;

/** array with a 'dense_dim' innermost dimension, and an arbitrary
 * shape otherwise, of rank 'Rank'. */
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

/** Copy the contents of the 'src' array or array_ref to the 'dest'
 * array or array_ref. The range of the shape of 'dest' will be copied,
 * and must be in bounds of 'src'. */
template <typename T, typename ShapeSrc, typename ShapeDest>
void copy(const array_ref<T, ShapeSrc>& src,
          const array_ref<typename std::remove_const<T>::type, ShapeDest>& dest) {
  typedef typename std::remove_const<T>::type non_const_T;
  internal::for_each_src_dest(src.data(), dest.data(), src.shape(), dest.shape(),
			      [&](T& src, non_const_T& dest) {
    dest = src;
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

/** Move the contents from the 'src' array or array_ref to the 'dest'
 * array or array_ref. The range of the shape of 'dest' will be moved,
 * and must be in bounds of 'src'. */
template <typename T, typename ShapeSrc, typename ShapeDest>
void move(const array_ref<T, ShapeSrc>& src, const array_ref<T, ShapeDest>& dest) {
  internal::for_each_src_dest(src.data(), dest.data(), src.shape(), dest.shape(),
			      [&](T& src, T& dest) {
    dest = std::move(src);
  });
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocDest>
void move(const array_ref<T, ShapeSrc>& src, array<T, ShapeDest, AllocDest>& dest) {
  move(src, dest.ref());
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc>
void move(array<T, ShapeSrc, AllocSrc>& src, const array_ref<T, ShapeDest>& dest) {
  move(src.ref(), dest);
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc, typename AllocDest>
void move(array<T, ShapeSrc, AllocSrc>& src, array<T, ShapeDest, AllocDest>& dest) {
  move(src.ref(), dest.ref());
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
  return make_copy(src.ref(), shape, alloc);
}

/** Make a copy of the 'src' array or array_ref with a new shape
 * 'shape'. The elements of 'src' are moved to the result. */
template <typename T, typename ShapeSrc, typename ShapeDest, typename Alloc = std::allocator<T>>
auto make_move(const array_ref<T, ShapeSrc>& src, const ShapeDest& shape,
               const Alloc& alloc = Alloc()) {
  array<T, ShapeDest, Alloc> dest(shape, alloc);
  move(src, dest);
  return dest;
}
template <typename T, typename ShapeSrc, typename ShapeDest, typename AllocSrc,
  typename AllocDest = AllocSrc>
auto make_move(array<T, ShapeSrc, AllocSrc>& src, const ShapeDest& shape,
               const AllocDest& alloc = AllocDest()) {
  return make_move(src.ref(), shape, alloc);
}

/** Make a copy of the 'src' array or array_ref with a dense shape of
 * the same rank as 'src'. */
template <typename T, typename ShapeSrc,
  typename Alloc = std::allocator<typename std::remove_const<T>::type>>
auto make_dense_copy(const array_ref<T, ShapeSrc>& src,
                     const Alloc& alloc = Alloc()) {
  return make_copy(src, make_dense(src.shape()), alloc);
}
template <typename T, typename ShapeSrc, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_dense_copy(const array<T, ShapeSrc, AllocSrc>& src,
                     const AllocDest& alloc = AllocDest()) {
  return make_dense_copy(src.ref(), alloc);
}

/** Make a copy of the 'src' array or array_ref with a dense shape of
 * the same rank as 'src'. The elements of 'src' are moved to the result. */
template <typename T, typename ShapeSrc, typename Alloc = std::allocator<T>>
auto make_dense_move(const array_ref<T, ShapeSrc>& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_dense(src.shape()), alloc);
}
template <typename T, typename ShapeSrc, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_dense_move(array<T, ShapeSrc, AllocSrc>& src, const AllocDest& alloc = AllocDest()) {
  return make_dense_move(src.ref(), alloc);
}

/** Make a copy of the 'src' array or array_ref with a compact version
 * of 'src's shape. */
template <typename T, typename Shape,
  typename Alloc = std::allocator<typename std::remove_const<T>::type>>
auto make_compact_copy(const array_ref<T, Shape>& src,
		       const Alloc& alloc = Alloc()) {
  return make_copy(src, make_compact(src.shape()), alloc);
}
template <typename T, typename Shape, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_compact_copy(const array<T, Shape, AllocSrc>& src,
		       const AllocDest& alloc = AllocDest()) {
  return make_compact_copy(src.ref(), alloc);
}

/** Make a copy of the 'src' array or array_ref with a compact version
 * of 'src's shape. The elements of 'src' are moved to the result. */
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
auto make_compact_move(const array_ref<T, Shape>& src, const Alloc& alloc = Alloc()) {
  return make_move(src, make_compact(src.shape()), alloc);
}
template <typename T, typename Shape, typename AllocSrc, typename AllocDest = AllocSrc>
auto make_compact_move(array<T, Shape, AllocSrc>& src, const AllocDest& alloc = AllocDest()) {
  return make_compact_move(src.ref(), alloc);
}

/** Reinterpret the array or array_ref 'a' of type 'T' to have a
 * different type 'U'. The size of 'T' must be equal to the size of
 * 'U'. */
template <typename U, typename T, typename Shape>
array_ref<U, Shape> reinterpret(const array_ref<T, Shape>& a) {
  static_assert(sizeof(T) == sizeof(U), "sizeof(reinterpreted type U) != sizeof(array type T)");
  return array_ref<U, Shape>(reinterpret_cast<U*>(a.data()), a.shape());
}
template <typename U, typename T, typename Shape, typename Alloc>
array_ref<U, Shape> reinterpret(array<T, Shape, Alloc>& a) {
  return reinterpret<U>(a.ref());
}
template <typename U, typename T, typename Shape, typename Alloc>
array_ref<const U, Shape> reinterpret(const array<T, Shape, Alloc>& a) {
  return reinterpret<const U>(a.ref());
}

/** Reshape the array or array_ref 'a' to have a new shape 'new_shape', with a
 * base pointer offset 'offset'. */
template <typename NewShape, typename T, typename OldShape>
array_ref<T, NewShape> reshape(const array_ref<T, OldShape>& a, NewShape new_shape, index_t offset = 0) {
  return array_ref<T, NewShape>(a.data() + offset, std::move(new_shape));
}
template <typename NewShape, typename T, typename OldShape, typename Allocator>
array_ref<T, NewShape> reshape(array<T, OldShape, Allocator>& a, NewShape new_shape, index_t offset = 0) {
  return reshape(a.ref(), new_shape, offset);
}
template <typename NewShape, typename T, typename OldShape, typename Allocator>
array_ref<const T, NewShape> reshape(const array<T, OldShape, Allocator>& a, NewShape new_shape, index_t offset = 0) {
  return reshape(a.ref(), new_shape, offset);
}

/** std::allocator-compatible Allocator that owns a buffer of fixed
 * size, which will be placed on the stack if the owning container is
 * allocated on the stack. This can only be used with containers that
 * have a maximum of one concurrent live allocation, which is the case
 * for array::array. */
// TODO: "stack_allocator" isn't a good name for this. It's a fixed
// allocation, but not necessarily a stack allocation.
template <class T, size_t N>
class stack_allocator {
  alignas(T) char buffer[N * sizeof(T)];
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
    if (allocated) ARRAY_THROW_BAD_ALLOC();
    if (n > N) ARRAY_THROW_BAD_ALLOC();
    allocated = true;
    return reinterpret_cast<T*>(&buffer[0]);
  }
  void deallocate(T* p, size_t) noexcept {
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

}  // namespace array

#endif  // ARRAY_ARRAY_H

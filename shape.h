#ifndef SHAPE_H
#define SHAPE_H

#include <cassert>
#include <tuple>
#include <functional>
#include <limits>

namespace array {

typedef std::size_t size_t;
typedef std::ptrdiff_t index_t;

enum : index_t {
  UNKNOWN = -1, //std::numeric_limits<index_t>::min(),
};

namespace internal {

// Given a compile-time static value, reconcile a compile-time
// static value and runtime value.
template <index_t Value>
index_t reconcile(index_t value) {
  if (Value != UNKNOWN) {
    // If we have a compile-time known value, ensure that it
    // matches the runtime value.
    assert(Value == value);
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
class index_iterator {
  index_t i;

 public:
  index_iterator(index_t i) : i(i) {}

  bool operator==(const index_iterator& r) const { return i == r.i; }
  bool operator!=(const index_iterator& r) const { return i != r.i; }

  index_t operator *() const { return i; }

  index_iterator operator++(int) { return index_iterator(i++); }
  index_iterator& operator++() { ++i; return *this; }
};
typedef index_iterator const_index_iterator;

/** Describes one dimension of an array. The template parameters
 * enable providing compile time constants for the min, extent, and
 * stride of the dim. */
template <index_t MIN = UNKNOWN, index_t EXTENT = UNKNOWN, index_t STRIDE = UNKNOWN>
class dim {
 protected:
  index_t min_;
  index_t extent_;
  index_t stride_;

 public:
  dim(index_t min = MIN, index_t extent = EXTENT, index_t stride = STRIDE)
    : min_(min), extent_(extent), stride_(stride) {}

  /** Index of the first element in this dim. */
  index_t min() const { return internal::reconcile<MIN>(min_); }
  /** Number of elements in this dim. */
  index_t extent() const { return internal::reconcile<EXTENT>(extent_); }
  /** Distance betwen elements in this dim. */
  index_t stride() const { return internal::reconcile<STRIDE>(stride_); }
  /** Index of the last element in this dim. */
  index_t max() const { return min() + extent() - 1; }

  /** Offset in memory of an index in this dim. */
  index_t offset(index_t at) const { return (at - min()) * stride(); }

  /** Check if the given index is within the range of this dim. */
  bool is_in_range(index_t at) const { return min() <= at && at <= max(); }

  /** Make an iterator referring to the first element in this dim. */
  const_index_iterator begin() const { return const_index_iterator(min()); }
  /** Make an iterator referring to one past the last element in this dim. */
  const_index_iterator end() const { return const_index_iterator(max() + 1); }

  bool operator==(const dim& other) const {
    return min() == other.min() && extent() == other.extent() && stride() == other.stride();
  }
  bool operator!=(const dim& other) const {
    return min() != other.min() || extent() != other.extent() || stride() != other.stride();
  }
};

/** A specialization of dim where the stride is known to be one. */
template <index_t MIN = UNKNOWN, index_t EXTENT = UNKNOWN>
using dense_dim = dim<MIN, EXTENT, 1>;

/** A specialization of dim where the stride is known to be zero. */
template <index_t MIN = UNKNOWN, index_t EXTENT = UNKNOWN>
using broadcast_dim = dim<MIN, EXTENT, 0>;

/** A dim where the indices wrap around outside the range of the dim. */
template <index_t EXTENT = UNKNOWN, index_t STRIDE = UNKNOWN>
class folded_dim {
 protected:
  index_t extent_;
  index_t stride_;
  
 public:
  folded_dim(index_t extent = EXTENT, index_t stride = STRIDE)
    : extent_(extent), stride_(stride) {}

  /** Non-folded range of the dim. */
  index_t extent() const { return internal::reconcile<EXTENT>(extent_); }
  /** Distance in memory between indices of this dim. */
  index_t stride() const { return internal::reconcile<STRIDE>(stride_); }

  /** In a folded dim, the min and max are unbounded. */
  index_t min() const { return std::numeric_limits<index_t>::min(); }
  index_t max() const { return std::numeric_limits<index_t>::max(); }

  /** Offset in memory of an element in this dim. */
  index_t offset(index_t at) const {
    return internal::euclidean_mod(at, extent()) * stride();
  }

  /** In a folded dim, all indices are in range. */
  bool is_in_range(index_t at) const { return true; }

  bool operator==(const folded_dim& other) const {
    return extent() == other.extent() && stride() == other.stride();
  }
  bool operator!=(const folded_dim& other) const {
    return extent() != other.extent() || stride() != other.stride();
  }
};

/** Clamp an index to the range described by a dim. */
template <typename Dim>
inline index_t clamp(index_t x, const Dim& d) {
  if (x < d.min()) return d.min();
  if (x > d.max()) return d.max();
  return x;
}

namespace internal {

// Base and general case for the sum of a variadic list of values.
inline index_t sum() {
  return 0;
}
inline index_t product() {
  return 1;
}

template <typename... Rest>
index_t sum(index_t first, Rest... rest) {
  return first + sum(rest...);
}
template <typename... Rest>
index_t product(index_t first, Rest... rest) {
  return first * product(rest...);
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

template <std::size_t Rank>
struct IndexType {
  typedef decltype(std::tuple_cat(std::make_tuple(index_t()), typename IndexType<Rank - 1>::type())) type;
};

template <>
struct IndexType<0> {
  typedef std::tuple<> type;
};

}  // namespace internal

/** A list of dims describing a multi-dimensional space of indices. */
template <typename... Dims>
class shape {
  std::tuple<Dims...> dims_;

 public:
  shape() {}
  shape(std::tuple<Dims...> dims) : dims_(std::move(dims)) {}
  shape(Dims... dims) : dims_(std::forward<Dims>(dims)...) {}

  /** Number of dims in this shape. */
  constexpr size_t rank() const { return std::tuple_size<std::tuple<Dims...>>::value; }

  typedef typename internal::IndexType<sizeof...(Dims)>::type index_type;

  /** Check if a list of indices is in range of this shape. */
  template <typename... Indices>
  bool is_in_range(const std::tuple<Indices...>& indices) const {
    return internal::is_in_range(dims_, indices);
  }
  template <typename... Indices>
  bool is_in_range(Indices... indices) const {
    return is_in_range(std::make_tuple<Indices...>(std::forward<Indices>(indices)...));
  }

  /** Compute the flat offset of the indices. If the indices are out
   * of range, throws std::out_of_range. */
  template <typename... Indices>
  index_t at(const std::tuple<Indices...>& indices) const {
    if (!is_in_range(indices)) {
      throw std::out_of_range("Indices are out of range");
    }
    return internal::flat_offset(dims_, std::make_tuple<Indices...>(std::forward<Indices>(indices)...));
  }
  template <typename... Indices>
  index_t at(Indices... indices) const {
    return at(std::make_tuple<Indices...>(std::forward<Indices>(indices)...));
  }

  /** Compute the flat offset of the indices. Does not check if the
   * indices are in range. */
  template <typename... Indices>
  index_t operator() (const std::tuple<Indices...>& indices) const {
    return internal::flat_offset(dims_, indices);
  }
  template <typename... Indices>
  index_t operator() (Indices... indices) const {
    return operator()(std::make_tuple<Indices...>(std::forward<Indices>(indices)...));
  }
  
  /** Get a specific dim of this shape. */
  template <std::size_t D>
  const auto& dim() const { return std::get<D>(dims_); }

  /** Compute the flat extent of this shape. This is one past the
   * range of possible values returned by at or operator(). */
  index_t flat_extent() const { return internal::flat_extent(dims_); }

  /** Compute the total number of items in the shape. */
  index_t size() const { return internal::product_of_extents(dims_); }

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

  bool operator==(const shape<>& other) const { return true; }
  bool operator!=(const shape<>& other) const { return false; }
};

namespace internal {

// This genius trick is from
// https://github.com/halide/Halide/blob/30fb4fcb703f0ca4db6c1046e77c54d9b6d29a86/src/runtime/HalideBuffer.h#L2088-L2108
template<size_t D, typename... Dims, typename Fn, typename... Indices,
  typename = decltype(std::declval<Fn>()(std::declval<Indices>()...))>
void for_all_indices_impl(int, const shape<Dims...>& shape, Fn &&f, Indices... indices) {
  f(indices...);
}

template<size_t D, typename... Dims, typename Fn, typename... Indices>
void for_all_indices_impl(double, const shape<Dims...>& shape, Fn &&f, Indices... indices) {
  for (index_t i : shape.template dim<D>()) {
    for_all_indices_impl<D - 1>(0, shape, std::forward<Fn>(f), i, indices...);
  }
}

template<size_t D, typename... Dims, typename Fn, typename... Indices,
  typename = decltype(std::declval<Fn>()(std::declval<std::tuple<Indices...>>()))>
void for_each_index_impl(int, const shape<Dims...>& shape, Fn &&f, const std::tuple<Indices...>& indices) {
  f(indices);
}

template<size_t D, typename... Dims, typename Fn, typename... Indices>
void for_each_index_impl(double, const shape<Dims...>& shape, Fn &&f, const std::tuple<Indices...>& indices) {
  for (index_t i : shape.template dim<D>()) {
    for_each_index_impl<D - 1>(0, shape, std::forward<Fn>(f), std::tuple_cat(std::make_tuple(i), indices));
  }
}

}  // namespace internal

/** Iterate over all indices in the shape, calling a function fn for
 * each set of indices. The indices are in the same order as the dims
 * in the shape. The first dim is the 'inner' loop of the iteration,
 * and the last dim is the 'outer' loop. */
template <typename... Dims, typename Fn>
void for_all_indices(const shape<Dims...>& s, const Fn& fn) {
  internal::for_all_indices_impl<sizeof...(Dims) - 1>(0, s, fn);
}

template <typename... Dims, typename Fn>
void for_each_index(const shape<Dims...>& s, const Fn& fn) {
  internal::for_each_index_impl<sizeof...(Dims) - 1>(0, s, fn, std::tuple<>());
} 

template <typename... Dims>
auto make_shape(Dims... dims) { return shape<Dims...>(std::forward<Dims>(dims)...); }

namespace internal {

inline std::tuple<> make_dims(index_t dim0_stride) {
  return std::tuple<>();
}

inline std::tuple<dim<>> make_dims(index_t dim0_stride, index_t dim0_extent) {
  dim<> dim0(0, dim0_extent, dim0_stride);
  return std::make_tuple(dim0);
}

template <typename... Extents>
auto make_dims(index_t dim0_stride, index_t dim0_extent, Extents... extents) {
  dim<> dim0(0, dim0_extent, dim0_stride);
  return std::tuple_cat(std::make_tuple(dim0), 
                        make_dims(dim0_extent * dim0_stride, std::forward<Extents>(extents)...));
}

template <typename... Dims>
auto make_shape_from_tuple(const std::tuple<Dims...>& dims) {
  return shape<Dims...>(dims);
}

}  // namespace internal

template <typename... Extents>
auto make_dense_shape(index_t dim0_extent, Extents... extents) {
  dense_dim<> dim0(0, dim0_extent);
  auto dims_tuple = std::tuple_cat(std::make_tuple(dim0), 
                                   internal::make_dims(dim0_extent, std::forward<Extents>(extents)...));
  return internal::make_shape_from_tuple(dims_tuple);
}

}  // namespace array

#endif

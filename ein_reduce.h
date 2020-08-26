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

/** \file ein_reduce.h
 * \brief Optional helper for computing Einstein reductions on arrays.
 */

#ifndef NDARRAY_EIN_REDUCE_H
#define NDARRAY_EIN_REDUCE_H

#include "array.h"

namespace nda {

namespace internal {

// TODO: Find a way to enable operations with non-op types? e.g. scalars.
template <class T>
using enable_if_ein_op =
    std::enable_if_t<std::is_same<typename T::is_ein_op, std::true_type>::value>;

template <class T>
using enable_if_ein_assign =
    std::enable_if_t<std::is_same<typename T::is_assign, std::true_type>::value>;

// Briefly, the high level goal of the next few classes is to enable construction of
// expressions describing Einstein summations or other reductions. This is done using
// a small expression template system. Normally, expression templates are troublesome
// due to overwhemling the compiler's ability to do CSE and other optimziations. In
// the case of Einstein reductions, the expressions will usually be very small...

template <class Derived>
struct ein_op_base {
  using is_ein_op = std::true_type;
  using is_assign = std::false_type;

  // We need to be able to get the derived type when creating binary operations using
  // this operation as an operand.
  const Derived& derived() const { return *static_cast<const Derived*>(this); }

  auto operator-() const { return make_ein_op_negate(derived()); }
  template <class T, class = enable_if_ein_op<T>>
  auto operator+(const T& r) const {
    return make_ein_op_add(derived(), r);
  }
  template <class T, class = enable_if_ein_op<T>>
  auto operator-(const T& r) const {
    return make_ein_op_sub(derived(), r);
  }
  template <class T, class = enable_if_ein_op<T>>
  auto operator*(const T& r) const {
    return make_ein_op_mul(derived(), r);
  }
  template <class T, class = enable_if_ein_op<T>>
  auto operator/(const T& r) const {
    return make_ein_op_div(derived(), r);
  }
};

// A leaf operand of an Einstein reduction expression. The Is... indicate the
// dimension of the reduction to use to address this operand.
template <class Op, size_t... Is>
struct ein_op : public ein_op_base<ein_op<Op, Is...>> {
  Op op;
  ein_op(Op op) : op(std::move(op)) {}

  // The largest dimension used by this operand.
  static constexpr index_t MaxIndex = sizeof...(Is) == 0 ? -1 : variadic_max(Is...);

  // auto doesn't work here because it doesn't include the reference type of operator() when we
  // need it, but it writing it includes it when we can't, e.g. if op(...) doesn't return a
  // reference.
  template <class Idx>
  NDARRAY_INLINE decltype(op(Is...)) operator()(const Idx& i) const {
    return op(std::get<Is>(i)...);
  }

  // Only ein_op has assignment operators.
  template <class T, class = enable_if_ein_op<T>>
  auto operator=(const T& r) const {
    return make_ein_op_assign(*this, r);
  }
  template <class T, class = enable_if_ein_op<T>>
  auto operator+=(const T& r) const {
    return make_ein_op_add_assign(*this, r);
  }
  template <class T, class = enable_if_ein_op<T>>
  auto operator-=(const T& r) const {
    return make_ein_op_sub_assign(*this, r);
  }
  template <class T, class = enable_if_ein_op<T>>
  auto operator*=(const T& r) const {
    return make_ein_op_mul_assign(*this, r);
  }
};

// A unary operation on an Einstein operand.
template <class Op, class Derived>
struct ein_unary_op : public ein_op_base<Derived> {
  Op op;
  ein_unary_op(const Op& op) : op(op) {}
  static constexpr index_t MaxIndex = Op::MaxIndex;
};

// Unary negate.
template <class Op>
struct ein_negate_op : public ein_unary_op<Op, ein_negate_op<Op>> {
  using base = ein_unary_op<Op, ein_negate_op<Op>>;
  ein_negate_op(const Op& op) : base(op) {}
  template <class Idx>
  NDARRAY_INLINE auto operator()(const Idx& i) const {
    return -base::op(i);
  }
};

template <class Op>
auto make_ein_op_negate(const Op& op) {
  return ein_negate_op<Op>(op);
}

// Cast to a different type.
template <class Type, class Op>
struct ein_cast_op : public ein_unary_op<Op, ein_cast_op<Type, Op>> {
  using base = ein_unary_op<Op, ein_cast_op<Type, Op>>;
  ein_cast_op(const Op& op) : base(op) {}
  template <class Idx>
  NDARRAY_INLINE auto operator()(const Idx& i) const {
    return static_cast<Type>(base::op(i));
  }
};

// A binary operation of two operands.
template <class OpA, class OpB, class Derived>
struct ein_binary_op : public ein_op_base<Derived> {
  OpA op_a;
  OpB op_b;
  ein_binary_op(const OpA& a, const OpB& b) : op_a(a), op_b(b) {}
  static constexpr index_t MaxIndex = std::max(OpA::MaxIndex, OpB::MaxIndex);
};

#define NDARRAY_MAKE_EIN_BINARY_HELPERS(name, op)                                                  \
  template <class OpA, class OpB>                                                                  \
  auto make_##name(const OpA& a, const OpB& b) {                                                   \
    return name<OpA, OpB>(a, b);                                                                   \
  }

#define NDARRAY_MAKE_EIN_BINARY_OP(name, op, is_assign_)                                           \
  template <class OpA, class OpB>                                                                  \
  struct name : public ein_binary_op<OpA, OpB, name<OpA, OpB>> {                                   \
    using base = ein_binary_op<OpA, OpB, name>;                                                    \
    name(const OpA& a, const OpB& b) : base(a, b) {}                                               \
    using is_assign = is_assign_;                                                                  \
    template <class Idx>                                                                           \
    NDARRAY_INLINE auto operator()(const Idx& i) const {                                           \
      return base::op_a(i) op base::op_b(i);                                                       \
    }                                                                                              \
  };                                                                                               \
  NDARRAY_MAKE_EIN_BINARY_HELPERS(name, op)

#define NDARRAY_MAKE_EIN_BINARY_FN(name, fn)                                                       \
  template <class OpA, class OpB>                                                                  \
  struct name : public ein_binary_op<OpA, OpB, name<OpA, OpB>> {                                   \
    using base = ein_binary_op<OpA, OpB, name>;                                                    \
    name(const OpA& a, const OpB& b) : base(a, b) {}                                               \
    template <class Idx>                                                                           \
    NDARRAY_INLINE auto operator()(const Idx& i) const {                                           \
      return fn(base::op_a(i), base::op_b(i));                                                     \
    }                                                                                              \
  };                                                                                               \
  NDARRAY_MAKE_EIN_BINARY_HELPERS(name, op)

// Define the expression types for the operations we support.
NDARRAY_MAKE_EIN_BINARY_OP(ein_op_add, +, std::false_type);
NDARRAY_MAKE_EIN_BINARY_OP(ein_op_sub, -, std::false_type);
NDARRAY_MAKE_EIN_BINARY_OP(ein_op_mul, *, std::false_type);
NDARRAY_MAKE_EIN_BINARY_OP(ein_op_div, /, std::false_type);
NDARRAY_MAKE_EIN_BINARY_FN(ein_op_min, std::min);
NDARRAY_MAKE_EIN_BINARY_FN(ein_op_max, std::max);

NDARRAY_MAKE_EIN_BINARY_OP(ein_op_assign, =, std::true_type);
NDARRAY_MAKE_EIN_BINARY_OP(ein_op_add_assign, +=, std::true_type);
NDARRAY_MAKE_EIN_BINARY_OP(ein_op_sub_assign, -=, std::true_type);
NDARRAY_MAKE_EIN_BINARY_OP(ein_op_mul_assign, *=, std::true_type);

#undef NDARRAY_MAKE_EIN_BINARY_FN
#undef NDARRAY_MAKE_EIN_BINARY_OP
#undef NDARRAY_MAKE_EIN_BINARY_HELPERS

} // namespace internal

/** Cast an Einstein summation operand to a different type `Type`.
 * The cast is performed using `static_cast<Type>`. */
template <class Type, class Op>
auto cast(const internal::ein_op_base<Op>& op) {
  return internal::ein_cast_op<Type, Op>(op.derived());
}

/** The min or max of two Einstein summation operands. */
template <class OpA, class OpB>
auto min(const internal::ein_op_base<OpA>& a, const internal::ein_op_base<OpB>& b) {
  return internal::make_ein_op_min(a.derived(), b.derived());
}
template <class OpA, class OpB>
auto max(const internal::ein_op_base<OpA>& a, const internal::ein_op_base<OpB>& b) {
  return internal::make_ein_op_max(a.derived(), b.derived());
}

namespace internal {

// Let these be found via ADL too.
using nda::cast;
using nda::max;
using nda::min;

// If multiple operands provide the same dim, we need to reconcile them
// to one dim. If you follow a compiler or runtime error here, your
// Einstein expression tries to address two dimensions that have different
// bounds with the same loop variable.
// TODO: It would be nice if this error would appear when constructing the
// Einstein expression when possible, but that's really hard to do.
template <class Dim0, class... Dims,
    class = std::enable_if_t<!any(not_equal(Dim0::Min, Dims::Min)...)>,
    class = std::enable_if_t<!any(not_equal(Dim0::Extent, Dims::Extent)...)>>
NDARRAY_INLINE const Dim0& reconcile_dim(const Dim0& dim0, const Dims&... dims) {
  if (dim0.stride() != 0) {
    // If the first dim is an output dimension, just require the other dims
    // to be in-bounds. This is a slightly relaxed requirement compared to the
    // compile-time constraints.
    assert(all(dims.is_in_range(dim0)...));
  } else {
    // If this is a reduction dimension, all of the dims must match. For
    // example, this is the constraint that checks that the inner dimensions of
    // a matrix multiplication are compatible.
    assert(all(dim0.min() == dims.min()...));
    assert(all(dim0.extent() == dims.extent()...));
  }
  return dim0;
}
// If we have zero dims, the user skipped a dim index, so we need a dummy
// loop.
NDARRAY_INLINE dim<0, 1, 0> reconcile_dim() { return {}; }

template <class... Dims, size_t... Is>
NDARRAY_INLINE auto reconcile_dim(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return reconcile_dim(std::get<Is>(dims)...);
}
template <class... Dims>
NDARRAY_INLINE auto reconcile_dim(const std::tuple<Dims...>& dims) {
  return reconcile_dim(dims, make_index_sequence<sizeof...(Dims)>());
}

// Get the shape of an ein_reduce operand, or an empty shape if not an array.
template <class T, class Shape>
NDARRAY_INLINE const auto& dims_of(const array_ref<T, Shape>& op) {
  return op.shape().dims();
}
template <class T>
NDARRAY_INLINE std::tuple<> dims_of(const T& op) {
  return std::tuple<>();
}

// Helper to reinterpret a dim with a new stride.
template <index_t NewStride, index_t Min, index_t Extent, index_t Stride>
NDARRAY_INLINE auto with_stride(const dim<Min, Extent, Stride>& d) {
  return dim<Min, Extent, NewStride>(d.min(), d.extent());
}
template <index_t NewStride, class Dim>
NDARRAY_INLINE auto with_stride(const std::tuple<Dim>& maybe_dim) {
  return std::make_tuple(with_stride<NewStride>(std::get<0>(maybe_dim)));
}
template <index_t NewStride>
NDARRAY_INLINE std::tuple<> with_stride(std::tuple<> maybe_dim) {
  return maybe_dim;
}

// These types are flags that let us overload behavior based on these 3 options.
class is_inferred_shape {};
class is_result_shape {};
class is_operand_shape {};

// Get a dim from an operand, depending on the intended use of the shape.
template <size_t Dim, class Dims, size_t... Is>
NDARRAY_INLINE auto gather_dims(is_result_shape, const ein_op<Dims, Is...>& op) {
  // If this is part of the result, we want to keep its strides.
  return get_or_empty<index_of<Dim, Is...>()>(dims_of(op.op));
}
template <size_t Dim, class Dims, size_t... Is>
NDARRAY_INLINE auto gather_dims(is_inferred_shape, const ein_op<Dims, Is...>& op) {
  // For inferred shapes, we want shapes without any constexpr strides, so it can be reshaped.
  return with_stride<dynamic>(get_or_empty<index_of<Dim, Is...>()>(dims_of(op.op)));
}
template <size_t Dim, class Dims, size_t... Is>
NDARRAY_INLINE auto gather_dims(is_operand_shape, const ein_op<Dims, Is...>& op) {
  // If this is an operand shape, we want all of its dimensions to be stride 0.
  return with_stride<0>(get_or_empty<index_of<Dim, Is...>()>(dims_of(op.op)));
}

template <size_t Dim, class Kind, class Op, class X>
NDARRAY_INLINE auto gather_dims(Kind kind, const ein_unary_op<Op, X>& op) {
  return gather_dims<Dim>(kind, op.op);
}
template <size_t Dim, class Kind, class OpA, class OpB, class X>
NDARRAY_INLINE auto gather_dims(Kind kind, const ein_binary_op<OpA, OpB, X>& op) {
  return std::tuple_cat(gather_dims<Dim>(kind, op.op_a), gather_dims<Dim>(kind, op.op_b));
}

template <size_t Dim, class Kind0, class Op0, class Kind1, class Op1>
NDARRAY_INLINE auto gather_dims(Kind0 kind0, const Op0& op0, Kind1 kind1, const Op1& op1) {
  return std::tuple_cat(gather_dims<Dim>(kind0, op0), gather_dims<Dim>(kind1, op1));
}
template <size_t... Is, class... KindAndOps>
NDARRAY_UNIQUE auto make_ein_reduce_shape(
    index_sequence<Is...>, const KindAndOps&... kind_and_ops) {
  return make_shape(reconcile_dim(gather_dims<Is>(kind_and_ops...))...);
}

} // namespace internal

/** Operand for an Einstein summation, which is an array or other
 * callable object, along with a set of dimension indices.
 * `ein<i, j, ...>(a)` means the dimensions `i, j, ...` of the
 * summation index are used to address `a` during Einstein
 * summation. The number of dimensions must match the number of
 * arguments of `a`. See `ein_reduce()` for more details. */
template <size_t... Is, class Op, class = internal::enable_if_callable<Op, decltype(Is)...>>
auto ein(Op op) {
  return internal::ein_op<Op, Is...>(std::move(op));
}
template <size_t... Is, class T, class Shape, class Alloc,
    class = std::enable_if_t<sizeof...(Is) == Shape::rank()>>
auto ein(array<T, Shape, Alloc>& op) {
  return ein<Is...>(op.ref());
}
template <size_t... Is, class T, class Shape, class Alloc,
    class = std::enable_if_t<sizeof...(Is) == Shape::rank()>>
auto ein(const array<T, Shape, Alloc>& op) {
  return ein<Is...>(op.ref());
}

/** Define an Einstein summation operand for a scalar. The scalar
 * is broadcasted as needed during the summation. Because this
 * operand does not provide a shape, the dimensions of the sum
 * must be inferred from other operands. See `ein_reduce()` for more
 * details. */
template <class T>
auto ein(T& scalar) {
  return ein<>(array_ref<T, shape<>>(&scalar, {}));
}

/** Compute an Einstein reduction. This function allows one to specify
 * many kinds of array transformations and reductions using
 * <a href="https://en.wikipedia.org/wiki/Einstein_notation">Einstein notation</a>.
 *
 * This function accepts an expression `expr` constructed using operators
 * on `ein<i, j, ...>(op)` operands. These operands describe which
 * dimensions of the reduction index should be used to address that
 * operand. The rank of the reduction operation is inferred from the
 * number of dimensions used in the expression. The reduction operation
 * is executed in the order of the indices, with the lowest numbered
 * dimension executed as the innermost loop.
 *
 * If `expr` is a reduction operator, the result must be initialized to
 * some useful value, typically the identity value for the reduction
 * operator, e.g. `0` for `+=`. Not initializing the result allows
 * successive `ein_reduce` operations to be applied to the same result.
 *
 * This function does not optimize the associative order in which the
 * operations are performed. It evaluates the expression for each element
 * of the final result reduction. This can be efficient for expansion
 * operations, but it may be inefficient for contractions. Contractions
 * may need to be reassociated manually for efficient computation.
 *
 * This function does not optimize the loop ordering within each operation.
 * The goal of this function is to provide a low-overhead and expressive
 * reduction that can be composed with other explicit loop transformations
 * to achieve good performance. Various optimization strategies can be
 * implemented by splitting loops appropriately and by controlling the
 * order of the loops with the reduction indices.
 *
 * Examples:
 * - `ein_reduce(ein<>(tr_A) += ein<i, i>(A))`, the trace of `A`.
 * - `ein_reduce(ein<>(dot) += (ein<i>(x) + ein<i>(y)) * ein<i>(z))`,
 *   the dot product `(x + y)*z`.
 * - `ein_reduce(ein<i, j>(AB) += ein<i, k>(A) * ein<k, j>(B))`, the matrix product `A*B`
 * - `ein_reduce(ein<i>(Ax) += ein<i, j>(A) * ein<j>(x))`, the matrix-vector product `A*x`
 * - `ein_reduce(ein<i>(diag_A) = ein<i, i>(A))`, the diagonal of `A`.
 *
 * where:
 * - `A`, `B`, `AB` are matrices (rank 2 arrays)
 * - `x`, `y`, `z`, `Ax` are vectors (rank 1 arrays)
 * - `tr_A`, `dot` are scalar (rank 0 arrays)
 * - `i`, `j`, `k` are unique constexpr integers. */
template <class Expr, class = internal::enable_if_ein_assign<Expr>>
NDARRAY_UNIQUE auto ein_reduce(const Expr& expr) {
  constexpr index_t LoopRank = Expr::MaxIndex + 1;

  // Gather the dimensions identified by the indices. gather_dims keeps the
  // first dimension it finds, so we want that to be the result dimension if it
  // is present. If not, this selects one of the operand dimensions, which are
  // given stride 0.
  auto reduction_shape = internal::make_ein_reduce_shape(internal::make_index_sequence<LoopRank>(),
      internal::is_result_shape(), expr.op_a, internal::is_operand_shape(), expr.op_b);

  // TODO: Try to compile-time optimize reduction_shape? :)
  // This is maybe actually somewhat doable, simply moving the for_each_index
  // call below into a function that could be overloaded based on the type of
  // the shape would enable different optimizations. This function could be a
  // member of a template parameter/parameter of this function, enabling the
  // caller to customize the optimization. However, it's still very difficult
  // to make useful optimizations without making some assumptions about the
  // dimensions of the shape.

  // Perform the reduction.
  for_each_index_in_order(reduction_shape, expr);

  // Assume the expr is an assignment, and return the left-hand side.
  return expr.op_a.op;
}

/** Infer the shape of the result of `make_ein_reduce`. */
template <size_t... ResultIs, class Expr, class = internal::enable_if_ein_op<Expr>>
auto make_ein_reduce_shape(const Expr& expr) {
  auto result_shape = internal::make_ein_reduce_shape(
      internal::index_sequence<ResultIs...>(), internal::is_inferred_shape(), expr);
  return make_compact(result_shape);
}

/** Compute an Einstein summation using `ein_reduce` and return the result. The
 * `value_type` of the result will be `T`, and the result shape will be inferred
 * from the shape of the operands. The result is initialized to `init` prior to
 * computing the summation. The Einstein summation indices for the result operand
 * are `ResultIs...`.
 *
 * Examples:
 * - `trace_A = make_ein_sum<T>(ein<i, i>(A))`
 * - `dot = make_ein_sum<T>((ein<i>(x) + ein<i>(y)) * ein<i>(z))`
 * - `AB = make_ein_sum<T, i, j>(ein<i, k>(A) * ein<k, j>(B))`
 * - `Ax = make_ein_sum<T, i>(ein<i, j>(A) * ein<1>(x))`
 *
 * where:
 * - `A`, `B` are matrices (rank 2 arrays)
 * - `x`, `y`, `z` are vectors (rank 1 arrays)
 * - `i`, `j`, `k` are unique constexpr integers.
 *
 * See `ein_reduce()` for more details.
 **/
// TODO: It would be nice to be able to express reductions other than sums.
// TODO: Add an overload with a default ResultIs... = 0, 1, 2, ... This requires
// also inferring the rank of the result.
template <class T, size_t... ResultIs, class Expr, class Alloc = std::allocator<T>,
    class = internal::enable_if_ein_op<Expr>>
NDARRAY_UNIQUE auto make_ein_sum(
    const Expr& expr, const T& init = T(), const Alloc& alloc = Alloc()) {
  auto result_shape = make_ein_reduce_shape<ResultIs...>(expr);
  auto result = make_array<T>(result_shape, init, alloc);
  ein_reduce(ein<ResultIs...>(result) += expr);
  return result;
}

} // namespace nda

#endif // NDARRAY_EIN_REDUCE_H

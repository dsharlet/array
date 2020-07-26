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

#ifndef NDARRAY_EINSUM_H
#define NDARRAY_EINSUM_H

#include "array.h"

namespace nda {

namespace internal {

template <class Arg, size_t... Is>
using einsum_arg = std::tuple<Arg, index_sequence<Is...>>;

// Make a dimension a reduction dimension (give it a constexpr stride 0).
template <index_t Min, index_t Extent, index_t Stride>
auto reduction(const dim<Min, Extent, Stride>& d) {
  return broadcast_dim<Min, Extent>(d.min(), d.extent());
}

// Make all of the dimensions reduction dimensions.
template <class... Dims, size_t... Is>
auto reductions(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return std::make_tuple(reduction(std::get<Is>(dims))...);
}
template <class... Dims>
auto reductions(const std::tuple<Dims...>& dims) {
  return reductions(dims, make_index_sequence<sizeof...(Dims)>());
}

// If a dim appears more than twice in gather_dims, the summation is ill-formed.
// TODO: Not sure about that... And even if it is true, we should generate a
// better error than failing here.
template <class Dim1>
auto reconcile_dim(const Dim1& dim1) { return dim1; }
template <class Dim1, class Dim2>
auto reconcile_dim(const Dim1& dim1, const Dim2& dim2) {
  assert(dim1.stride() != 0 || dim2.stride() != 0 || dim1.min() == dim2.min());
  assert(dim1.stride() != 0 || dim2.stride() != 0 || dim1.max() == dim2.max());
  return dim1;
}

template <class... Dims, size_t... Is>
auto reconcile_dim(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return reconcile_dim(std::get<Is>(dims)...);
}
template <class... Dims>
auto reconcile_dim(const std::tuple<Dims...>& dims) {
  return reconcile_dim(dims, make_index_sequence<sizeof...(Dims)>());
}

// Gather all of the dimensions for einsum arguments into one shape.
template <size_t Dim, size_t... Is, class Dims>
auto gather_dim(const einsum_arg<Dims, Is...>& arg) {
  return get_tuple<index_of<Dim, Is...>()>(std::get<0>(arg));
}
template <size_t Dim, class... Args>
auto gather_dims(const Args&... args) {
  return reconcile_dim(std::tuple_cat(gather_dim<Dim>(args)...));
}
template <class... Dims, size_t... Is>
auto gather_dims(index_sequence<Is...>, const Dims&... dims) {
  return std::make_tuple(gather_dims<Is>(dims...)...);
}

// Infer the dims of the result of an einsum.
template <class... Dims, size_t... Is>
auto infer_result_dim(const std::tuple<Dims...>& dims, index_sequence<Is...>) {
  return reconcile_dim(std::get<Is>(dims)...);
}
template <class... Dims>
auto infer_result_dim(const std::tuple<Dims...>& dims) {
  return reconcile_dim(dims, make_index_sequence<sizeof...(Dims)>());
}

template <size_t Dim, class... Args>
auto infer_result_dims(const Args&... args) {
  return infer_result_dim(std::tuple_cat(gather_dim<Dim>(args)...));
}
template <class... Dims, size_t... Is>
auto infer_result_dims(index_sequence<Is...>, const Dims&... dims) {
  return std::make_tuple(infer_result_dims<Is>(dims...)...);
}

// Call operator() on an einsum argument, using the einsum indices as a shuffle.
template <class Idx, class Arg, size_t... Is>
NDARRAY_INLINE auto ein_at(const einsum_arg<Arg, Is...>& ein, const Idx& i) {
  return std::get<0>(ein)(std::get<Is>(i)...);
}

template <class T>
NDARRAY_INLINE T product() { return static_cast<T>(1); }
template <class T, class... Ts>
NDARRAY_INLINE T product(T a, Ts... b) { return a * product<T>(b...); }

// TODO: FIgure out how to compute LoopRank from Args. It should be quite
// doable, but all of my attempts either hit constexpr issues or
// cause clang to hang (!!).
template <size_t LoopRank, class... Args, class Result>
void einsum_impl(const Result& result, const Args&... args) {
  const auto& result_dims = std::get<0>(result).shape().dims();

  // Gather the dimensions identified by the indices. gather_dims keeps the
  // first dimension it finds, so we want that to be the result dimension if it
  // is present. If not, this selects one of the argument dimensions, which are
  // given stride 0.
  auto reduction_shape = make_shape_from_tuple(gather_dims(
      make_index_sequence<LoopRank>(),
      std::make_tuple(result_dims, std::get<1>(result)),
      std::make_tuple(reductions(std::get<0>(args).shape().dims()), std::get<1>(args))...));

  // TODO: Try to compile-time optimize reduction_shape? :)

  // Reinterpret the result as having a shape of the reduction dimensions.
  auto reduction = reinterpret_shape(std::get<0>(result), reduction_shape);

  for_each_index(reduction_shape, [&](const index_of_rank<LoopRank>& i) {
    reduction(i) += product(ein_at(args, i)...);
  });
}

// Figure out the shape of the result of an einsum.
template <size_t... ResultIs, class... Args>
auto infer_einsum_result_shape(const Args&... args) {
  return make_shape_from_tuple(infer_result_dims(
    make_index_sequence<sizeof...(ResultIs)>(),
    std::make_tuple(std::get<0>(args).shape().dims(), std::get<1>(args))...));
}

}  // namespace internal

/** Argument for an Einstein summation, which is an array along with
 * a set of dimension indices. `ein<i, j, ...>(a)` means the dimensions
 * `i, j, ...` of the summation index are used to address `a` during
 * Einstein summation. See `einsum` for more details. */
template <size_t... Is, class T, class Shape,
    class = std::enable_if_t<sizeof...(Is) == Shape::rank()>>
auto ein(const array_ref<T, Shape>& op) {
  return std::make_tuple(op, internal::index_sequence<Is...>());
}
template <size_t... Is, class T, class Shape, class Alloc,
    class = std::enable_if_t<sizeof...(Is) == Shape::rank()>>
auto ein(array<T, Shape, Alloc>& op) {
  return ein<Is...>(op.ref());
}
template <size_t... Is, class T, class Shape, class Alloc,
    class = std::enable_if_t<sizeof...(Is) == Shape::rank()>>
auto ein(const array<T, Shape, Alloc>& op) {
  return ein<Is...>(op.cref());
}

/** Compute an Einstein summation. This function allows one to specify
 * many kinds of array transformations and reductions using Einstein
 * notation. See https://en.wikipedia.org/wiki/Einstein_notation for more
 * information about the notation itself.
 *
 * This function accepts a list of arguments arg0, ..., result.
 * Each argument is the result of the `ein<i, j, ...>(arg)` helper
 * function, which describes which dimensions of the summation index
 * should be used to address that argument.
 *
 * The result of the summation is added to `result`. `result` must be
 * initialized to some useful value (typically 0) before calling this
 * function.
 *
 * Examples:
 * - `einsum(ein<0, 0>(A), ein<>(tr_A))`, the trace of A.
 * - `einsum(ein<0>(x), ein<0>(y), ein<>(dot_xy))`, the dot product x*y.
 * - `einsum(ein<0, 2>(A), ein<2, 1>(B), einsum<0, 1>(AB))`, the matrix product A*B
 * - `einsum(ein<0, 1>(A), ein<1>(x), ein<0>(Ax))`, the matrix-vector product A*x
 *
 * where:
 * - `A`, `B`, `AB` are matrices (rank 2 arrays)
 * - `x`, `y`, `Ax` are vectors (rank 1 arrays)
 * - `tr_A`, `dot_xy` are scalar (rank 0 arrays)
 **/
template <
    class Arg0, size_t... Arg0Is, class Arg1, size_t... Arg1Is, class ResultArg, size_t... ResultIs>
void einsum(
    const internal::einsum_arg<Arg0, Arg0Is...>& arg0,
    const internal::einsum_arg<Arg1, Arg1Is...>& arg1,
    const internal::einsum_arg<ResultArg, ResultIs...>& result) {
  constexpr size_t LoopRank = internal::variadic_max(Arg0Is..., Arg1Is..., ResultIs...) + 1;

  internal::einsum_impl<LoopRank>(result, arg0, arg1);
}
template <size_t... Arg0Is, size_t... ResultIs, class Arg0, class ResultArg>
void einsum(
    const internal::einsum_arg<Arg0, Arg0Is...>& arg0,
    const internal::einsum_arg<ResultArg, ResultIs...>& result) {
  constexpr size_t LoopRank = internal::variadic_max(Arg0Is..., ResultIs...) + 1;

  internal::einsum_impl<LoopRank>(result, arg0);
}

/** Infer the shape of the result of `make_einsum`. */
template <size_t... ResultIs, class... Args>
auto make_einsum_shape(const Args&... args) {
  auto result_shape = internal::infer_einsum_result_shape<ResultIs...>(args...);
  // TODO: This would really benefit from addressing https://github.com/dsharlet/array/issues/31
  return make_compact(result_shape);
}

/** Compute an Einstein summation and return the result. The `value_type` of the
 * result will be `T`, and the shape will be inferred from the shape of the
 * operands. The result is initialized to 0 prior to computing the summation.
 * The Einstein summation indices for the result are `ResultIs...`.
 *
 * Examples:
 * - `tr(A) = make_einsum<T>(ein<0, 0>(A))`
 * - `dot(x, y) = make_einsum<T>(ein<0>(x), ein<0>(y))`
 * - `A*B = make_einsum<T, 0, 1>(ein<0, 2>(A), ein<2, 1>(B))`
 * - `A*x = make_einsum<T, 0>(ein<0, 1>(A), ein<1>(x))`
 *
 * where:
 * - `A`, `B` are matrices (rank 2 arrays)
 * - `x`, `y` are vectors (rank 1 arrays)
 **/
// TODO: Add an overload with a default ResultIs... = 0, 1, 2, ... This requires
// also inferring the rank of the result.
template <
    class T, size_t... ResultIs, class Arg0, size_t... Arg0Is, class Arg1, size_t... Arg1Is,
    class Alloc = std::allocator<T>>
auto make_einsum(
    const internal::einsum_arg<Arg0, Arg0Is...>& arg0,
    const internal::einsum_arg<Arg1, Arg1Is...>& arg1,
    const Alloc& alloc = Alloc()) {
  auto result_shape = make_einsum_shape<ResultIs...>(arg0, arg1);
  auto result = make_array<T>(result_shape, static_cast<T>(0), alloc);
  einsum(arg0, arg1, ein<ResultIs...>(result));
  return result;
}
template <
    class T, size_t... ResultIs, class Arg0, size_t... Arg0Is,
    class Alloc = std::allocator<T>>
auto make_einsum(
    const internal::einsum_arg<Arg0, Arg0Is...>& arg0,
    const Alloc& alloc = Alloc()) {
  auto result_shape = make_einsum_shape<ResultIs...>(arg0);
  auto result = make_array<T>(result_shape, static_cast<T>(0), alloc);
  einsum(arg0, ein<ResultIs...>(result));
  return result;
}

// TODO: Consider supporting einsum of more than 2 operands.

}  // namespace nda

#endif  // NDARRAY_IMAGE_H

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

/** Argument for an Einstein summation, which is an array along with
 * a set of dimension indices. */
template <class Arg, size_t... Is>
using einsum_arg = std::tuple<Arg, std::index_sequence<Is...>>;
template <size_t... Is, class Arg>
einsum_arg<Arg, Is...> ein(const Arg& op) {
  return std::make_tuple(op, std::index_sequence<Is...>());
}

namespace internal {

// Make a dimension a reduction dimension (give it a constexpr stride 0).
template <index_t Min, index_t Extent, index_t Stride>
auto reduction(const dim<Min, Extent, Stride>& d) {
  return dim<Min, Extent, 0>(d.min(), d.extent());
}

// Make all of the dimensions reduction dimensions.
template <class... Dims, size_t... Is>
auto reductions(const std::tuple<Dims...>& dims, std::index_sequence<Is...>) {
  return std::make_tuple(reduction(std::get<Is>(dims))...);
}
template <class... Dims>
auto reductions(const std::tuple<Dims...>& dims) {
  return reductions(dims, std::make_index_sequence<sizeof...(Dims)>());
}

// If a dim appears other than twice in gather_dims, the summation is ill-formed.
// TODO: Not sure about that...
template <class Dim1, class Dim2>
auto reconcile_dim(const Dim1& dim1, const Dim2& dim2) {
  assert(dim1.min() == dim2.min());
  assert(dim1.max() == dim2.max());
  return dim1;
}

template <class... Dims, size_t... Is>
auto reconcile_dim(const std::tuple<Dims...>& dims, std::index_sequence<Is...>) {
  return reconcile_dim(std::get<Is>(dims)...);
}
template <class... Dims>
auto reconcile_dim(const std::tuple<Dims...>& dims) {
  return reconcile_dim(dims, std::make_index_sequence<sizeof...(Dims)>());
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
auto gather_dims(std::index_sequence<Is...>, const Dims&... dims) {
  return std::make_tuple(gather_dims<Is>(dims...)...);
}

// Call operator() on an einsum argument, using the einsum indices as a shuffle.
template <class Idx, class Arg, size_t... Is>
auto ein_at(const einsum_arg<Arg, Is...>& ein, const Idx& i) {
  return std::get<0>(ein)(std::get<Is>(i)...);
}

}  // namespace internal

template <
    size_t... Arg1Is, size_t... ResultIs,
    class Arg1, class ResultArg>
void einsum(
    const einsum_arg<Arg1, Arg1Is...>& arg1,
    const einsum_arg<ResultArg, ResultIs...>& result) {
  constexpr size_t LoopRank = internal::variadic_max(Arg1Is..., ResultIs...) + 1;

  const auto& result_dims = std::get<0>(result).shape().dims();

  // Dimensions we take from the operands are reductions, i.e. they should
  // have stride 0.
  const auto& arg1_dims = internal::reductions(std::get<0>(arg1).shape().dims());

  // Gather the dimensions identified by the indices. gather_dims keeps the
  // first dimension, so we want that to be the result dimension if it is
  // present. If not, this selects one of the argument dimensions, which will
  // have stride 0.
  auto reduction_shape = make_shape_from_tuple(internal::gather_dims(
      std::make_index_sequence<LoopRank>(),
      std::make_tuple(result_dims, std::get<1>(result)),
      std::make_tuple(arg1_dims, std::get<1>(arg1))));

  // TODO: Try to compile-time optimize reduction_shape :)

  // Reinterpret the result as having a shape of the reduction dimensions.
  auto reduction = reinterpret_shape(std::get<0>(result), reduction_shape);

  for_each_index(reduction_shape, [&](const index_of_rank<LoopRank>& i) {
    reduction(i) += internal::ein_at(arg1, i);
  });
}

template <
    size_t... Arg1Is, size_t... Arg2Is, size_t... ResultIs,
    class Arg1, class Arg2, class ResultArg>
void einsum(
    const einsum_arg<Arg1, Arg1Is...>& arg1,
    const einsum_arg<Arg2, Arg2Is...>& arg2,
    const einsum_arg<ResultArg, ResultIs...>& result) {
  constexpr size_t LoopRank = internal::variadic_max(Arg1Is..., Arg2Is..., ResultIs...) + 1;

  const auto& result_dims = std::get<0>(result).shape().dims();

  // Dimensions we take from the operands are reductions, i.e. they should
  // have stride 0.
  const auto& arg1_dims = internal::reductions(std::get<0>(arg1).shape().dims());
  const auto& arg2_dims = internal::reductions(std::get<0>(arg2).shape().dims());

  // Gather the dimensions identified by the indices. gather_dims keeps the
  // first dimension, so we want that to be the result dimension if it is
  // present. If not, this selects one of the argument dimensions, which will
  // have stride 0.
  auto reduction_shape = make_shape_from_tuple(internal::gather_dims(
      std::make_index_sequence<LoopRank>(),
      std::make_tuple(result_dims, std::get<1>(result)),
      std::make_tuple(arg1_dims, std::get<1>(arg1)),
      std::make_tuple(arg2_dims, std::get<1>(arg2))));

  // TODO: Try to compile-time optimize reduction_shape :)

  // Reinterpret the result as having a shape of the reduction dimensions.
  auto reduction = reinterpret_shape(std::get<0>(result), reduction_shape);

  for_each_index(reduction_shape, [&](const index_of_rank<LoopRank>& i) {
    reduction(i) += internal::ein_at(arg1, i) * internal::ein_at(arg2, i);
  });
}

}  // namespace nda

#endif  // NDARRAY_IMAGE_H

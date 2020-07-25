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

/** \file image.h
 * \brief Optional image-specific helpers and specializations.
*/
#ifndef NDARRAY_EINSUM_H
#define NDARRAY_EINSUM_H

#include "array.h"

namespace nda {

template <class Arg, size_t... Is>
using einsum_arg = std::tuple<Arg, std::index_sequence<Is...>>;

template <size_t... Is, class Arg>
einsum_arg<Arg, Is...> ein(const Arg& op) {
  return {op, std::index_sequence<Is...>()};
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

// Given a list of dimensions, assert they have all equal min and extent,
// and return the first.
template <class Dim>
auto reconcile_dim(const Dim& dim) {
  return dim;
}
template <class Dim1, class... Dims>
auto reconcile_dim(const Dim1& dim1, const Dims&... dims) {
  assert(all(dim1.min() == dims.min()...));
  assert(all(dim1.max() == dims.max()...));
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
template <
    size_t Dim, size_t... Arg1Is, size_t... Arg2Is, size_t... Arg3Is,
    class Dims1, class Dims2, class Dims3>
auto gather_dim(
    const einsum_arg<Dims1, Arg1Is...>& arg1,
    const einsum_arg<Dims2, Arg2Is...>& arg2,
    const einsum_arg<Dims3, Arg3Is...>& arg3) {
  return reconcile_dim(std::tuple_cat(
      get_tuple<index_of<Dim, Arg1Is...>()>(std::get<0>(arg1)),
      get_tuple<index_of<Dim, Arg2Is...>()>(std::get<0>(arg2)),
      get_tuple<index_of<Dim, Arg3Is...>()>(std::get<0>(arg3))));
}
template <class... Dims, size_t... Is>
auto gather_dims(std::index_sequence<Is...>, const Dims&... dims) {
  return std::make_tuple(gather_dim<Is>(dims...)...);
}

}  // namespace internal

template <
    size_t... Arg1Is, size_t... Arg2Is, size_t... ResultIs,
    class Arg1, class Arg2, class T, class Shape>
void einsum(
    const einsum_arg<Arg1, Arg1Is...>& arg1,
    const einsum_arg<Arg2, Arg2Is...>& arg2,
    const einsum_arg<array_ref<T, Shape>, ResultIs...>& result) {

  constexpr size_t LoopRank = internal::variadic_max(Arg1Is..., Arg2Is..., ResultIs...) + 1;

  const auto& result_dims = std::get<0>(result).shape().dims();

  // Dimensions we take from the operands are reductions, i.e. they should
  // have stride 0.
  const auto& arg1_dims = internal::reductions(std::get<0>(arg1).shape().dims());
  const auto& arg2_dims = internal::reductions(std::get<0>(arg2).shape().dims());

  // Gather the dimensions identified by the indices.
  auto reduction_shape = make_shape_from_tuple(internal::gather_dims(
      std::make_index_sequence<LoopRank>(),
      std::make_tuple(result_dims, std::get<1>(result)),
      std::make_tuple(arg1_dims, std::get<1>(arg1)),
      std::make_tuple(arg2_dims, std::get<1>(arg2))));

  // Reinterpret the result as having a shape of the reduction dimensions.
  auto reduction = reinterpret_shape(std::get<0>(result), reduction_shape);

  auto op1 = std::get<0>(arg1);
  auto op2 = std::get<0>(arg2);

  for_each_index(reduction_shape, [&](const index_of_rank<LoopRank>& i) {
    reduction(i) += op1(std::get<Arg1Is>(i)...) * op2(std::get<Arg2Is>(i)...);
  });
}


}  // namespace nda

#endif  // NDARRAY_IMAGE_H

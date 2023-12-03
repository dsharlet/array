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

/** \file z_order.h
 * \brief Helpers for traversing multi-dimensional ranges in z-order.
 */
#ifndef NDARRAY_Z_ORDER_H
#define NDARRAY_Z_ORDER_H

#include "array.h"

#include <algorithm>
#include <functional>

namespace nda {

namespace internal {

inline index_t next_power_of_two(index_t x) { 
  index_t result = 1;
  while (result < x) {
    result <<= 1;
  }
  return result;
}

// TODO: This probably doesn't need fn to be inlined, and it would reduce template instantiations
// to use std::function instead.
template <size_t Rank, class Fn>
NDARRAY_UNIQUE void for_each_index_in_z_order_impl(const std::array<index_t, Rank>& end,
    std::array<index_t, Rank> z, index_t dim, index_t step, const Fn& fn) {
  if (dim == 0 && step == 1) {
    // We're at the innermost step of the traversal, call fn for the two neighboring points we have.
    fn(z);
    z[0] += 1;
    if (z[0] < end[0]) {
      fn(z);
    }
  } else if (dim == 0) {
    // We're on the innermost dimension, but not the innermost step.
    // Move to the next step for the outermost dimension.
    for_each_index_in_z_order_impl(end, z, Rank - 1, step >> 1, fn);
    z[0] += step;
    if (z[0] < end[0]) {
      for_each_index_in_z_order_impl(end, z, Rank - 1, step >> 1, fn);
    }
  } else {
    // Move to the next dimension for the same step.
    for_each_index_in_z_order_impl(end, z, dim - 1, step, fn);
    z[dim] += step;
    if (z[dim] < end[dim]) {
      for_each_index_in_z_order_impl(end, z, dim - 1, step, fn);
    }
  }
}

// TODO: If this conformed more to the rest of the array API, perhaps it could be
// exposed as a user API.
template <class Extents, class Fn>
NDARRAY_UNIQUE void for_each_index_in_z_order(const Extents& extents, const Fn& fn) {
  constexpr index_t Rank = std::tuple_size<Extents>::value;
  // Get the ends of the iteration space as an array.
  const auto end = tuple_to_array<index_t>(extents, make_index_sequence<Rank>());
  const index_t max_extent = *std::max_element(end.begin(), end.end());
  const index_t step = std::max<index_t>(1, next_power_of_two(max_extent) >> 1);
  std::array<index_t, Rank> z = {{0,}};
  for_each_index_in_z_order_impl(end, z, Rank - 1, step, fn);
}

template <class... Ranges, class Fn, size_t... Is>
NDARRAY_UNIQUE void for_each_in_z_order_impl(
    const std::tuple<Ranges...>& ranges, const Fn& fn, index_sequence<Is...>) {
  constexpr size_t Rank = sizeof...(Is);
  std::array<index_t, Rank> extents = {
      {(std::end(std::get<Is>(ranges)) - std::begin(std::get<Is>(ranges)))...}};
  for_each_index_in_z_order(extents, [&](const std::array<index_t, Rank>& i) {
    fn(std::make_tuple(*(std::begin(std::get<Is>(ranges)) + std::get<Is>(i))...));
  });
}

} // namespace internal

/** Iterate over a multi-dimensional iterator range in "z-order", by following a
 * z-order curve. This ordering may be useful for optimizing for locality. The
 * iterators must be random access iterators.
 */
template <class... Ranges, class Fn>
NDARRAY_UNIQUE void for_each_in_z_order(
    const std::tuple<Ranges...>& ranges, const Fn& fn) {
  internal::for_each_in_z_order_impl(
      ranges, fn, internal::make_index_sequence<sizeof...(Ranges)>());
}
template <class Fn, class... Ranges>
NDARRAY_UNIQUE void for_all_in_z_order(
    const std::tuple<Ranges...>& ranges, const Fn& fn) {
  internal::for_each_in_z_order_impl(
      ranges, [&](const auto& i) { internal::apply(fn, i); },
      internal::make_index_sequence<sizeof...(Ranges)>());
}

} // namespace nda

#endif // NDARRAY_Z_ORDER_H

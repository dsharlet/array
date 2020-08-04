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

/** \file matrix.h
 * \brief Optional matrix-specific helpers and specializations.
 */
#ifndef NDARRAY_MATRIX_H
#define NDARRAY_MATRIX_H

#include "array.h"

namespace nda {

/** The standard matrix notation is to refer to elements by 'row,
 * column'. To make this efficient for typical programs, we're going
 * to make the second dimension the dense dim. This shape has the
 * option of making the size of the matrix compile-time constant via
 * the template parameters. */
template <index_t Rows = dynamic, index_t Cols = dynamic>
using matrix_shape = shape<dim<dynamic, Rows>, dense_dim<dynamic, Cols>>;

/** A matrix or matrix_ref is an `array` or `array_ref` with `Shape` =
 * `matrix_shape`. */
template <class T, index_t Rows = dynamic, index_t Cols = dynamic, class Alloc = std::allocator<T>>
using matrix = array<T, matrix_shape<Rows, Cols>, Alloc>;
template <class T, index_t Rows = dynamic, index_t Cols = dynamic>
using matrix_ref = array_ref<T, matrix_shape<Rows, Cols>>;
template <class T, index_t Rows = dynamic, index_t Cols = dynamic>
using const_matrix_ref = matrix_ref<const T, Rows, Cols>;

/** A vector is just a 1-d array. */
template <index_t Length = dynamic>
using vector_shape = shape<dense_dim<dynamic, Length>>;

template <class T, index_t Length = dynamic, class Alloc = std::allocator<T>>
using vector = array<T, vector_shape<Length>, Alloc>;
template <class T, index_t Length = dynamic>
using vector_ref = array_ref<T, vector_shape<Length>>;
template <class T, index_t Length = dynamic>
using const_vector_ref = vector_ref<const T, Length>;

/** A matrix with static dimensions `Rows` and `Cols`, with an
 * `auto_allocator`. */
template <class T, index_t Rows, index_t Cols>
using small_matrix = array<T, matrix_shape<Rows, Cols>, auto_allocator<T, Rows * Cols>>;
template <class T, index_t Length>
using small_vector = array<T, vector_shape<Length>, auto_allocator<T, Length>>;

/** Calls `fn` for each index in a matrix shape `s`. */
template <class Shape, class Fn>
void for_each_matrix_index(const Shape& s, Fn&& fn) {
  for (index_t i : s.i()) {
    for (index_t j : s.j()) {
      fn(std::tuple<index_t, index_t>(i, j));
    }
  }
}

template <index_t Rows, index_t Cols>
class shape_traits<matrix_shape<Rows, Cols>> {
public:
  typedef matrix_shape<Rows, Cols> shape_type;

  template <class Fn>
  static void for_each_index(const shape_type& s, Fn&& fn) {
    for_each_matrix_index(s, fn);
  }

  template <class Ptr, class Fn>
  static void for_each_value(const shape_type& s, Ptr base, Fn&& fn) {
    for_each_matrix_index(
        s, [=, fn = std::move(fn)](const typename shape_type::index_type& i) { fn(base[s(i)]); });
  }
};

} // namespace nda

#endif // NDARRAY_MATRIX_H
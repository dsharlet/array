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
#ifndef NDARRAY_IMAGE_H
#define NDARRAY_IMAGE_H

#include "array.h"

namespace nda {

/** A generic image is any 3D array with dimensions x, y, c. c represents the
 * channels of the image, typically it will have extent 3 or 4, with red, green,
 * and blue mapped to indices in this dimension. */
using image_shape = shape_of_rank<3>;
template <typename T>
using image = array_of_rank<T, 3>;
template <typename T>
using image_ref = array_ref_of_rank<T, 3>;

/** A 'chunky' image is an array with 3 dimensions x, y, c, where c is dense,
 * and the dimension with the next stride is x. The stride in x may be larger
 * than the number of channels, to allow for padding pixels to a convenient
 * alignment. This is a common image storage format used by many programs
 * working with images. */
template <index_t Channels, index_t ChannelStride = Channels>
using chunky_image_shape =
  shape<strided_dim<ChannelStride>, dim<>, dense_dim<0, Channels>>;
template <typename T, index_t Channels, index_t ChannelStride = Channels>
using chunky_image = array<T, chunky_image_shape<Channels, ChannelStride>>;
template <typename T, index_t Channels, index_t ChannelStride = Channels>
using chunky_image_ref = array_ref<T, chunky_image_shape<Channels, ChannelStride>>;

/** Calls 'fn' for each index in an image shape 's'. c is the innermost
 * dimension of the loop nest. */
template <typename Shape, typename Fn>
void for_each_image_index(const Shape& s, Fn&& fn) {
  // Images should always be iterated with c as the innermost loop. Even when
  // the image is planar, the number of channels is generally small, and many
  // operations use all of the channels at the same time (all-to-all
  // communication in the c dimension).
  for (index_t y : s.y()) {
    for (index_t x : s.x()) {
      for (index_t c : s.c()) {
        fn(std::make_tuple(x, y, c));
      }
    }
  }
}

template <index_t Channels, index_t ChannelStride>
class shape_traits<chunky_image_shape<Channels, ChannelStride>> {
 public:
  typedef chunky_image_shape<Channels, ChannelStride> shape_type;

  template <typename Fn>
  static void for_each_index(const shape_type& s, Fn&& fn) {
    for_each_image_index(s, fn);
  }

  template <typename Fn, typename... T>
  static void for_each_value(const shape_type& s, Fn&& fn, T... base) {
    for_each_image_index(s, [=, &fn](const typename shape_type::index_type& i) {
      index_t offset = s(i);
      fn(base[offset]...);
    });
  }
};

template <index_t Channels>
class shape_traits<chunky_image_shape<Channels>> {
 public:
  typedef chunky_image_shape<Channels> shape_type;

  template <typename Fn>
  static void for_each_index(const shape_type& s, Fn&& fn) {
    for_each_image_index(s, fn);
  }

  // When Channels == ChannelStride, we can implement for_each_value by fusing
  // the x and c dimensions.
  template <typename Fn, typename... T>
  static void for_each_value(const shape_type& s, Fn&& fn, T... base) {
    dense_shape<2> opt_s({s.x().min() * Channels, s.x().extent() * Channels}, s.y());
    for_each_value_in_order(opt_s, fn, base...);
  }
};

/** A 'planar' image is an array with dimensions x, y, c, where x is dense. This
 * format is less common, but more convenient for optimization, particularly
 * SIMD vectorization. Note that this shape also supports 'line-chunky' storage
 * orders. */
using planar_image_shape = dense_shape<3>;
template <typename T>
using planar_image = dense_array<T, 3>;
template <typename T>
using planar_image_ref = dense_array_ref<T, 3>;

enum class crop_origin {
  /** The result of the crop has min 0, 0. */
  zero,
  /** The result indices inside the crop are the same as the original indices,
   * and the result indices outside the crop are out of bounds. */
  crop,
};

/** Crop an image shape 's' to the indices [x0, x1) x [y0, y1). The origin of
 * the new shape is determined by 'origin'. */
template <typename Shape>
Shape crop_image_shape(Shape s, index_t x0, index_t y0, index_t x1, index_t y1,
                       crop_origin origin = crop_origin::crop) {
  s.x().set_extent(x1 - x0);
  s.y().set_extent(y1 - y0);
  switch (origin) {
  case crop_origin::zero:
    s.x().set_min(0);
    s.y().set_min(0);
    break;
  case crop_origin::crop:
    s.x().set_min(x0);
    s.y().set_min(y0);
    break;
  }
  return s;
}

/** Crop the 'im' image or image ref to the range [x0, x1) x [y0, y1). The
 * result is a ref of the input image. The origin of the result is determined by
 * 'origin'. */
template <typename T, typename Shape>
array_ref<T, Shape> crop(const array_ref<T, Shape>& im,
                         index_t x0, index_t y0, index_t x1, index_t y1,
                         crop_origin origin = crop_origin::crop) {
  Shape cropped_shape = crop_image_shape(im.shape(), x0, y0, x1, y1, origin);
  index_t c0 = im.shape().c().min();
  T* base = im.base() != nullptr ? &im(x0, y0, c0) : nullptr;
  if (origin == crop_origin::crop) {
    base = internal::pointer_add(base, -cropped_shape(x0, y0, c0));
  }
  return array_ref<T, Shape>(base, cropped_shape);
}
template <typename T, typename Shape>
array_ref<const T, Shape> crop(const array<T, Shape>& im,
                               index_t x0, index_t y0, index_t x1, index_t y1,
                               crop_origin origin = crop_origin::crop) {
  return crop(im.ref(), x0, y0, x1, y1, origin);
}
template <typename T, typename Shape>
array_ref<T, Shape> crop(array<T, Shape>& im,
                         index_t x0, index_t y0, index_t x1, index_t y1,
                         crop_origin origin = crop_origin::crop) {
  return crop(im.ref(), x0, y0, x1, y1, origin);
}

/** Get a 2-dimensional ref of the 'Channel' channel of the 'im' image or image
 * ref. */
template <typename T, typename Shape>
auto slice_channel(const array_ref<T, Shape>& im, index_t channel) {
  auto shape = reorder<0, 1>(im.shape());
  T* base = im.base() != nullptr ? &im(im.x().min(), im.y().min(), channel) : nullptr;
  return array_ref<T, decltype(shape)>(base, shape);
}
template <typename T, typename Shape>
auto slice_channel(const array<T, Shape>& im, index_t channel) {
  return slice_channel(im.ref(), channel);
}
template <typename T, typename Shape>
auto slice_channel(array<T, Shape>& im, index_t channel) {
  return slice_channel(im.ref(), channel);
}

}  // namespace nda

#endif  // NDARRAY_IMAGE_H

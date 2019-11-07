#ifndef ARRAY_IMAGE_H
#define ARRAY_IMAGE_H

#include "array.h"

namespace array {

/** A generic image is any 3D array with dimensions x, y, c. c
 * represents the channels of the image, typically it will have
 * extent 3 or 4, with red, green, and blue mapped to indices in
 * this dimension. */
using image_shape = shape_of_rank<3>;
template <typename T>
using image = array_of_rank<T, 3>;
template <typename T>
using image_ref = array_ref_of_rank<T, 3>;

/** A 'chunky' image is an array with 3 dimensions x, y, c, where c is
 * dense, and the dimension with the next stride is x. This is a
 * common image storage format used by many programs working with
 * images. */
template <index_t Channels>
using chunky_image_shape =
  shape<strided_dim<Channels>, dim<>, dense_dim<0, Channels>>;
template <typename T, index_t Channels>
using chunky_image = array<T, chunky_image_shape<Channels>>;
template <typename T, index_t Channels>
using chunky_image_ref = array_ref<T, chunky_image_shape<Channels>>;

template <index_t Channels>
class shape_traits<chunky_image_shape<Channels>> {
 public:
  template <typename Fn>
  static void for_each_index(const chunky_image_shape<Channels>& s, Fn&& fn) {
    // chunky images should be iterated on in this order due to
    // c being dense.
    for (index_t y : s.y()) {
      for (index_t x : s.x()) {
	for (index_t c : s.c()) {
	  std::forward<Fn>(fn)(std::make_tuple(x, y, c));
	}
      }
    }
  }

  // We want for_each_value to just use for_each_index on this shape.
  using optimize_for_each_value_at_runtime = std::false_type;
};

/** A 'planar' iamge is an array with dimensions x, y, c, where x is
 * dense. This format is less common, but more convenient for
 * optimization, particularly SIMD vectorization. Note that this
 * shape also supports 'line-chunky' storage orders. */
using planar_image_shape = dense_shape<3>;
template <typename T>
using planar_image = dense_array<T, 3>;
template <typename T>
using planar_image_ref = dense_array_ref<T, 3>;

enum class crop_origin {
  /** The result of the crop has min 0, 0. */
  zero,
  /** The result indices inside the crop are the same as the original
   * indices, and the result indices outside the crop are out of
   * bounds. */
  crop,
};

/** Crop an image shape 's' to the indices [x0, x1) x [y0, y1). The
 * origin of the new shape is determined by 'origin'. */
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

/** Crop the 'im' image or image ref to the range of indices [x0, x1)
 * x [y0, y1). The result is a ref of the input image. The origin of
 * the result is determined by 'origin'. */
template <typename T, typename Shape>
array_ref<T, Shape> crop(const array_ref<T, Shape>& im,
			 index_t x0, index_t y0, index_t x1, index_t y1,
			 crop_origin origin = crop_origin::crop) {
  Shape cropped_shape = crop_image_shape(im.shape(), x0, y0, x1, y1, origin);
  index_t c0 = im.shape().c().min();
  T* base = &im(x0, y0, c0);
  if (origin == crop_origin::crop) {
    base -= cropped_shape(x0, y0, c0);
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

/** Get a 2-dimensional ref of the 'Channel' channel of the 'im' image
 * or image ref. */
template <index_t Channel, typename T, typename Shape>
auto slice_channel(const array_ref<T, Shape>& im) {
  auto shape = permute<0, 1>(im.shape());
  T* base = &im(im.x().min(), im.y().min(), Channel);
  return array_ref<T, decltype(shape)>(base, shape);
}
template <index_t Channel, typename T, typename Shape>
auto slice_channel(const array<T, Shape>& im) {
  return slice_channel<Channel>(im.ref());
}
template <index_t Channel, typename T, typename Shape>
auto slice_channel(array<T, Shape>& im) {
  return slice_channel<Channel>(im.ref());
}

}  // namespace array

#endif

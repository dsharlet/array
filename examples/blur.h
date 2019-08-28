#ifndef ARRAY_EXAMPLES_BLUR_H
#define ARRAY_EXAMPLES_BLUR_H

#include "array.h"

// Blur 5 integer samples with a 1 4 6 4 1 kernel.
template <typename T>
T gaussian_blur_5(T x0, T x1, T x2, T x3, T x4) {
  return (x0 + x1*4 + x2*6 + x3*4 + x4 + 8)/16;
}

// 2D blur an image with a 1 4 6 4 1 kernel in each dimension.
template <typename Image>
void gaussian_blur_5x5(const Image& in, Image& out) {
  auto x_dim = out.template dim<0>();
  auto y_dim = out.template dim<1>();
  auto c_dim = out.template dim<2>();
  // The strategy here is to blur each row in y first, then blur that
  // in x, and do this per line of the output.
  // This means we need to store one line of the blur in y, so let's
  // define a shape of a buffer for a single line.
  auto line_buffer_shape = make_shape(x_dim, c_dim);
  auto blur_y = array::make_array<typename Image::value_type>(line_buffer_shape);
  for (int y : y_dim) {
    // Blur in y first.
    for (int x : x_dim) {
      for (int c : c_dim) {
        auto y0 = in(x, clamp(y - 2, y_dim), c);
        auto y1 = in(x, clamp(y - 1, y_dim), c);
        auto y2 = in(x, clamp(y, y_dim), c);
        auto y3 = in(x, clamp(y + 1, y_dim), c);
        auto y4 = in(x, clamp(y + 2, y_dim), c);
        blur_y(x, c) = gaussian_blur_5(y0, y1, y2, y3, y4);
      }
    }

    // Blur in x.
    for (int x : x_dim) {
      for (int c : c_dim) {
        auto x0 = blur_y(clamp(x - 2, x_dim), c);
        auto x1 = blur_y(clamp(x - 1, x_dim), c);
        auto x2 = blur_y(clamp(x, x_dim), c);
        auto x3 = blur_y(clamp(x + 1, x_dim), c);
        auto x4 = blur_y(clamp(x + 2, x_dim), c);
        out(x, y, c) = gaussian_blur_5(x0, x1, x2, x3, x4);
      }
    }
  }
}

#endif  // ARRAY_EXAMPLES_BLUR_H

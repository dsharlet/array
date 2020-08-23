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

#ifndef NDARRAY_RESAMPLE_H
#define NDARRAY_RESAMPLE_H

#include "array.h"
#include "ein_reduce.h"
#include "rational.h"

#include <cmath>
#include <functional>

namespace nda {

/** A reconstruction kernel is a continuous function. */
using continuous_kernel = std::function<float(float)>;

/** Box kernel. */
inline float box(float s) { return std::abs(s) <= 0.5f ? 1.0f : 0.0f; }

/** Linear interpolation kernel. */
inline float linear(float s) { return std::max(0.0f, 1.0f - std::abs(s)); }

// The quadratic and cubic formulas come from
// https://pdfs.semanticscholar.org/45e0/92c057ffe242665ef44590f2b8d725696d76.pdf
inline float quadratic_family(float s, float r) {
  s = std::abs(s);
  float s2 = s * s;
  if (s < 0.5f) {
    return -2.0f * r * s2 + 0.5f * (r + 1.0f);
  } else if (s < 1.5f) {
    return r * s2 + (-2.0f * r - 0.5f) * s + 0.75f * (r + 1);
  } else {
    return 0;
  }
}

inline float cubic_family(float s, float B, float C) {
  s = std::abs(s);
  float s2 = s * s;
  float s3 = s2 * s;
  if (s <= 1.0f) {
    return (2.0f - B * 1.5f - C) * s3 + (-3.0f + 2.0f * B + C) * s2 + (1.0f - B / 3.0f);
  } else if (s <= 2.0f) {
    return (-B / 6.0f - C) * s3 + (B + 5.0f * C) * s2 + (-2.0f * B - 8.0f * C) * s +
           (B * 4.0f / 3.0f + 4.0f * C);
  } else {
    return 0.0f;
  }
}

/** Interpolating quadratic kernel. */
inline float interpolating_quadratic(float s) { return quadratic_family(s, 1.0f); }

/** Interpolating cubic, i.e. Catmull-Rom spline. */
inline float interpolating_cubic(float s) { return cubic_family(s, 0.0f, 0.5f); }

/** Smooth but soft quadratic B-spline approximation (not interpolating). */
inline float quadratic_bspline(float s) { return quadratic_family(s, 0.5f); }

/** Smooth but soft cubic B-spline approximation (not interpolating). */
inline float cubic_bspline(float s) { return cubic_family(s, 1.0f, 0.0f); }

inline float sinc(float s) {
  s *= M_PI;
  return std::abs(s) > 1e-6f ? std::sin(s) / s : 1.0f;
}

/** Lanczos kernel, with `2*side_lobes + 1` lobes. */
inline float lanczos(float s, int side_lobes) {
  if (std::abs(s) <= side_lobes) {
    return sinc(s) * sinc(s / side_lobes);
  } else {
    return 0.0f;
  }
}
template <int side_lobes>
float lanczos(float s) {
  return lanczos(s, side_lobes);
}

namespace internal {

// An array of kernels is not just a 2D array, because each kernel may
// have different bounds.
using kernel_allocator = auto_allocator<float, 16>;
using kernel_array = dense_array<dense_array<float, 1, kernel_allocator>, 1>;

// Build kernels for each index in a dim 'out' to sample from a dim 'in'.
// The kernels are guaranteed not to read out of bounds of 'in'.
inline kernel_array build_kernels(
    interval<> in, interval<> out, const rational<index_t>& rate, continuous_kernel kernel) {
  // The constant 1/2 as a rational.
  const rational<index_t> half = rational<index_t>(1, 2);

  // We need to compute a kernel for each output position.
  kernel_array kernels(out);

  // Define a buffer to produce each kernel in.
  dense_array<float, 1> buffer(in);

  // When downsampling, stretch the kernel to perform low pass filtering.
  // TODO: Move this, so it's possible to specify kernels that include
  // low pass filtering, e.g. trapezoid kernels.
  float kernel_scale = std::min(to_float(rate), 1.0f);

  for (index_t x : out) {
    // Compute the fractional position of the input corresponding to
    // this output.
    const float in_x = to_float((x + half) / rate - half);

    // Fill the buffer, while keeping track of the sum of, first, and
    // last non-zero kernel values,
    // TODO: This is messy, optimize it better.
    // TODO: This might produce incorrect results if a kernel has zeros mixed
    // in with non-zeros before the "end" (though such kernels probably aren't
    // very good).
    index_t min = in.max();
    index_t max = in.min();
    float sum = 0.0f;
    for (index_t rx : in) {
      float k_rx = kernel((rx - in_x) * kernel_scale);
      buffer(rx) = k_rx;
      if (k_rx != 0.0f) {
        sum += k_rx;
        min = std::min(min, rx);
        max = std::max(max, rx);
      }
    }

    // Crop and normalize the kernel.
    index_t extent = max - min + 1;
    assert(extent > 0);
    assert(sum > 0.0f);
    dense_array<float, 1, kernel_allocator> kernel_x({{min, extent}});
    for (index_t rx : kernel_x.x()) {
      kernel_x(rx) = buffer(rx) / sum;
    }

    // Make a copy without the padding, and store it in the kernel array.
    kernels(x) = std::move(kernel_x);
  }

  return kernels;
}

// Resize the y dimension of an input array 'in' to a destination array 'out',
// using kernels(y) to produce out(., y, .).
template <class TIn, class TOut>
void resample_y(const TIn& in, const TOut& out, const kernel_array& kernels) {
  enum { x = 0, ry = 1, c = 2 };
  for (index_t y : out.y()) {
    const auto& kernel_y = kernels(y);
    fill(out(_, y, _), 0.0f);
    // TODO: Consider making reconcile_dim in ein_reduce take the intersection
    // of the dims to avoid needing the crop of in here.
    ein_reduce(
        ein<x, c>(out(_, y, _)) += ein<x, ry, c>(in(_, kernel_y.x(), _)) * ein<ry>(kernel_y));
  }
}

// TODO: Get rid of these ugly helpers. Shapes shouldn't preserve strides in some usages.
template <index_t Min, index_t Extent, index_t Stride>
dim<Min, Extent> without_stride(const dim<Min, Extent, Stride>& d) {
  return dim<Min, Extent>(d.min(), d.extent());
}

template <class X, class Y, class C>
auto make_temp_image_shape(X x, Y y, C c) {
  return make_shape(with_stride<1>(x), without_stride(y), without_stride(c));
}

template <class T, class X, class Y, class C>
auto make_temp_image(X x, Y y, C c) {
  return make_array<T>(make_temp_image_shape(x, y, c));
}

} // namespace internal

/** Resample an array `in` to produce an array `out`, using an interpolation `kernel`.
 * Input coordinates (x, y) map to output coordinates (x * rate_x, y * rate_y). */
template <class TIn, class TOut, class ShapeIn, class ShapeOut>
void resample(array_ref<TIn, ShapeIn> in, array_ref<TOut, ShapeOut> out, rational<index_t> rate_x,
    rational<index_t> rate_y, continuous_kernel kernel) {
  // Make the kernels we need at each output x and y coordinate in the output.
  internal::kernel_array kernels_x = internal::build_kernels(
      {in.x().min(), in.x().extent()}, {out.x().min(), out.x().extent()}, rate_x, kernel);
  internal::kernel_array kernels_y = internal::build_kernels(
      {in.y().min(), in.y().extent()}, {out.y().min(), out.y().extent()}, rate_y, kernel);

  // Split the image into horizontal strips.
  constexpr index_t StripSize = 64;
  for (auto yo : split<StripSize>(out.y())) {
    auto out_y = out(out.x(), yo, out.c());

    // Resample the input in y, to an intermediate buffer.
    auto strip = internal::make_temp_image<TOut>(in.x(), out_y.y(), out_y.c());
    internal::resample_y(in, strip.ref(), kernels_y);

    // Transpose the intermediate.
    enum { x = 0, y = 1, c = 2 };
    auto strip_tr = internal::make_temp_image<TOut>(out_y.y(), in.x(), out_y.c());
    ein_reduce(ein<x, y, c>(strip_tr) = ein<y, x, c>(strip));

    // Resample the intermediate in x.
    auto out_tr = internal::make_temp_image<TOut>(out_y.y(), out_y.x(), out_y.c());
    internal::resample_y(strip_tr.cref(), out_tr.ref(), kernels_x);

    // Transpose the intermediate to the output.
    ein_reduce(ein<x, y, c>(out_y) = ein<y, x, c>(out_tr));
  }
}

} // namespace nda

#endif // NDARRAY_RESAMPLE_H

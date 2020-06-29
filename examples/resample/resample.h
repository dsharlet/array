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

#ifndef RESAMPLE_H
#define RESAMPLE_H

#include "array.h"
#include "rational.h"

#include <cmath>
#include <functional>

// A reconstruction kernel is a continuous function.
using continuous_kernel = std::function<float(float)>;

// Define some common kernels useful for resizing images.
inline float box(float s) {
  return std::abs(s) <= 0.5f ? 1.0f : 0.0f;
}

inline float linear(float s) {
  return std::max(0.0f, 1.0f - std::abs(s));
}

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
    return
        (2.0f - B * 1.5f - C) * s3 +
        (-3.0f + 2.0f * B + C) * s2 +
        (1.0f - B / 3.0f);
  } else if (s <= 2.0f) {
    return
        (-B / 6.0f - C) * s3 +
        (B + 5.0f * C) * s2 +
        (-2.0f * B - 8.0f * C) * s +
        (B * 4.0f / 3.0f + 4.0f * C);
  } else {
    return 0.0f;
  }
}

// An interpolating quadratic.
inline float interpolating_quadratic(float s) {
  return quadratic_family(s, 1.0f);
}

// An interpolating cubic, i.e. Catmull-Rom spline.
inline float interpolating_cubic(float s) {
  return cubic_family(s, 0.0f, 0.5f);
}

// Smooth but soft quadratic B-spline approximation.
inline float quadratic_bspline(float s) {
  return quadratic_family(s, 0.5f);
}

// Smooth but soft cubic B-spline approximation.
inline float cubic_bspline(float s) {
  return cubic_family(s, 1.0f, 0.0f);
}

inline float sinc(float s) {
  s *= M_PI;
  return std::abs(s) > 1e-6f ? std::sin(s) / s : 1.0f;
}

inline float lanczos(float s, int lobes) {
  if (std::abs(s) <= lobes) {
    return sinc(s) * sinc(s / lobes);
  } else {
    return 0.0f;
  }
}

template<int lobes>
float lanczos(float s) { return lanczos(s, lobes); }

namespace internal {

// An array of kernels is not just a 2D array, because each kernel may
// have different bounds.
using kernel_array = nda::dense_array<nda::dense_array<float, 1>, 1>;

// Build kernels for each index in a dim 'out' to sample from a dim 'in'.
// The kernels are guaranteed not to read out of bounds of 'in'.
inline kernel_array build_kernels(
    nda::dim<> in, nda::dim<> out, const rational<nda::index_t>& rate,
    continuous_kernel kernel) {
  // The constant 1/2 as a rational.
  const rational<nda::index_t> half = rational<nda::index_t>(1, 2);

  // We need to compute a kernel for each output position.
  kernel_array kernels(make_dense(make_shape(out)));

  // Define a buffer to produce each kernel in.
  nda::dense_array<float, 1> buffer(make_dense(make_shape(in)));

  // When downsampling, stretch the kernel to perform low pass filtering.
  // TODO: Move this, so it's possible to specify kernels that include
  // low pass filtering, e.g. trapezoid kernels.
  float kernel_scale = std::min(to_float(rate), 1.0f);

  for (nda::index_t x : out) {
    // Compute the fractional position of the input corresponding to
    // this output.
    const float in_x = to_float((x + half) / rate - half);

    // Fill the buffer, while keeping track of the sum of, first, and
    // last non-zero kernel values,
    // TODO: This is messy, optimize it better.
    // TODO: This might produce incorrect results if a kernel has zeros mixed
    // in with non-zeros before the "end" (though such kernels probably aren't
    // very good).
    nda::index_t min = in.max();
    nda::index_t max = in.min();
    float sum = 0.0f;
    for (nda::index_t rx : in) {
      float k_rx = kernel((rx - in_x) * kernel_scale);
      buffer(rx) = k_rx;
      if (k_rx != 0.0f) {
        sum += k_rx;
        min = std::min(min, rx);
        max = std::max(max, rx);
      }
    }

    // Crop and normalize the kernel.
    nda::index_t extent = max - min + 1;
    assert(extent > 0);
    assert(sum > 0.0f);
    nda::dense_array<float, 1> kernel_x(nda::dense_shape<1>(nda::dense_dim<>(min, extent)));
    for (nda::index_t rx : kernel_x.x()) {
      kernel_x(rx) = buffer(rx) / sum;
    }

    // Make a copy without the padding, and store it in the kernel array.
    kernels(x) = std::move(kernel_x);
  }

  return kernels;
}

// Resize the y dimension of an input array 'in' to a destination array 'out',
// using kernels(y) to produce out(., y, .).
template <typename TIn, typename TOut>
void resample_y(const TIn& in, const TOut& out, const kernel_array& kernels) {
  for (nda::index_t y : out.y()) {
    const nda::dense_array<float, 1>& kernel_y = kernels(y);
    for (nda::index_t c : out.c()) {
      for (nda::index_t x : out.x()) {
        out(x, y, c) = 0.0f;
      }
      for (nda::index_t ry : kernel_y.x()) {
        for (nda::index_t x : out.x()) {
          out(x, y, c) += in(x, ry, c) * kernel_y(ry);
        }
      }
    }
  }
}

template <typename TIn, typename TOut>
void transpose(const TIn& in, const TOut& out) {
  for (nda::index_t c : out.c()) {
    for (nda::index_t y : out.y()) {
      for (nda::index_t x : out.x()) {
        out(x, y, c) = in(y, x, c);
      }
    }
  }
}

}  // namespace internal

template <typename TIn, typename TOut, typename ShapeIn, typename ShapeOut>
void resample(const nda::array_ref<TIn, ShapeIn>& in, const nda::array_ref<TOut, ShapeOut>& out,
            const rational<nda::index_t>& rate_x, const rational<nda::index_t>& rate_y,
            continuous_kernel kernel) {
  internal::kernel_array kernels_x =
      internal::build_kernels(in.x(), out.x(), rate_x, kernel);
  internal::kernel_array kernels_y =
      internal::build_kernels(in.y(), out.y(), rate_y, kernel);

  constexpr nda::index_t StripSize = 64;
  for (nda::index_t y = out.y().min(); y <= out.y().max(); y += StripSize) {
    auto out_y = crop(out, out.x().min(), y, out.x().max() + 1, y + StripSize);
    nda::planar_image<TOut> strip(make_dense(make_shape(in.x(), out_y.y(), out.c())));
    nda::planar_image<TOut> strip_tr(make_dense(make_shape(out_y.y(), in.x(), out.c())));
    nda::planar_image<TOut> out_tr(make_dense(make_shape(out_y.y(), out.x(), out.c())));

    internal::resample_y(in, strip.ref(), kernels_y);
    internal::transpose(strip.cref(), strip_tr.ref());
    internal::resample_y(strip_tr.cref(), out_tr.ref(), kernels_x);
    internal::transpose(out_tr.cref(), out_y);
  }
}

#endif

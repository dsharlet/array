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

#include "array.h"
#include "image.h"
#include "benchmark.h"
#include "rational.h"

#include <random>
#include <iostream>
#include <functional>

#include <Magick++.h>

using namespace nda;

// An array of kernels is not just a 2D array, because each kernel may
// have different bounds.
using kernel_array = dense_array<dense_array<float, 1>, 1>;

// The constant 1/2 as a rational.
const rational<index_t> half = rational<index_t>(1, 2);

// Build kernels for each index in a dim 'out' to sample from a dim 'in'.
// The kernels are guaranteed not to read out of bounds of 'in'.
kernel_array build_kernels(
    dim<> in, dim<> out, const rational<index_t>& rate,
    std::function<float(float)> kernel) {
  // We need to compute a kernel for each output position.
  kernel_array kernels(make_dense(make_shape(out)));

  // Define a buffer to produce each kernel in.
  dense_array<float, 1> buffer(make_dense(make_shape(in)));

  // When downsampling, stretch the kernel to perform low pass filtering.
  float kernel_scale = std::min(to_float(rate), 1.0f);

  for (index_t out_x : out) {
    // Compute the fractional position of the input corresponding to
    // this output.
    const float in_x = to_float((out_x + half) / rate - half);

    // Fill the buffer, while keeping track of the sum of, first, and
    // last non-zero kernel values,
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
    dense_array_ref<float, 1> cropped_kernel(&buffer(min), dense_dim<>(min, extent));
    for (index_t ry : cropped_kernel.x()) {
      cropped_kernel(ry) /= sum;
    }

    // Make a copy without the padding, and store it in the kernel array.
    kernels(out_x) = make_dense_copy(cropped_kernel);
  }

  return kernels;
}

// Resize the y dimension of an input array 'in' to a destination array 'out',
// using kernels(y) to produce out(., y, .).
template <typename TIn, typename TOut>
void resize_y(const TIn& in, const TOut& out, const kernel_array& kernels) {
  for (index_t y : out.y()) {
    dense_array<float, 1> kernel_y = kernels(y);
    for (index_t c : out.c()) {
      for (index_t x : out.x()) {
        out(x, y, c) = 0.0f;
      }
      for (index_t ry : kernel_y.x()) {
        for (index_t x : out.x()) {
          out(x, y, c) += in(x, ry, c) * kernel_y(ry);
        }
      }
    }
  }
}

template <typename TIn, typename TOut>
void transpose(const TIn& in, const TOut& out) {
  for (index_t c : out.c()) {
    for (index_t y : out.y()) {
      for (index_t x : out.x()) {
        out(x, y, c) = in(y, x, c);
      }
    }
  }
}

template <typename TIn, typename TOut, typename ShapeIn, typename ShapeOut>
void resize(const array_ref<TIn, ShapeIn>& in, const array_ref<TOut, ShapeOut>& out,
            const rational<index_t>& rate_x, const rational<index_t>& rate_y,
            std::function<float(float)> kernel) {
  kernel_array kernels_x = build_kernels(in.x(), out.x(), rate_x, kernel);
  kernel_array kernels_y = build_kernels(in.y(), out.y(), rate_y, kernel);

  constexpr index_t StripSize = 64;
  for (index_t y = out.y().min(); y <= out.y().max(); y += StripSize) {
    auto out_y = crop(out, out.x().min(), y, out.x().max() + 1, y + StripSize);
    planar_image<TOut> strip(make_dense(make_shape(in.x(), out_y.y(), out.c())));
    planar_image<TOut> strip_tr(make_dense(make_shape(out_y.y(), in.x(), out.c())));
    planar_image<TOut> out_tr(make_dense(make_shape(out_y.y(), out.x(), out.c())));

    resize_y(in, strip.ref(), kernels_y);
    transpose(strip.cref(), strip_tr.ref());
    resize_y(strip_tr.cref(), out_tr.ref(), kernels_x);
    transpose(out_tr.cref(), out_y);
  }
}

// Define some common kernels useful for resizing images.
float nearest(float s) {
  return -0.5f <= s && s < 0.5f ? 1.0f : 0.0f;
}

float linear(float s) {
  return std::max(0.0f, 1.0f - std::abs(s));
}

// The quadratic and cubic formulas come from
// https://pdfs.semanticscholar.org/45e0/92c057ffe242665ef44590f2b8d725696d76.pdf
float quadratic_family(float s, float r) {
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

// An interpolating quadratic.
float quadratic(float s) {
  return quadratic_family(s, 1.0f);
}

// A C1-continuous quadratic.
float quadratic_C1(float s) {
  return quadratic_family(s, 0.5f);
}

float cubic_family(float s, float B, float C) {
  s = std::abs(s);
  float s2 = s * s;
  float s3 = s2 * s;
  if (s <= 1.0f) {
    return (2.0f - B * 1.5f - C) * s3 + (-3.0f + 2.0f * B + C) * s2 + (1 - B / 3.0f);
  } else if (s <= 2.0f) {
    return (-B / 6.0f - C) * s3 + (B + 5.0f * C) * s2 + (-2.0f * B - 8.0f * C) * s + (B * 4.0f / 3.0f + 4.0f * C);
  } else {
    return 0.0f;
  }
}

float approximate_bspline(float s) {
  return cubic_family(s, 1.0f, 0.0f);
}

float catmullrom(float s) {
  return cubic_family(s, 0.0f, 0.5f);
}

float sinc(float s) {
  s *= M_PI;
  return std::abs(s) > 1e-6f ? std::sin(s) / s : 1.0f;
}

float lanczos(float s, int lobes) {
  if (-lobes <= s && s < lobes) {
    return sinc(s) * sinc(s / lobes);
  } else {
    return 0.0f;
  }
}
template<int lobes>
float lanczos(float s) { return lanczos(s, lobes); }


std::function<float(float)> parse_kernel(const char* name) {
  if (strcmp(name, "nearest") == 0) {
    return nearest;
  } else if (strcmp(name, "linear") == 0) {
    return linear;
  } else if (strcmp(name, "quadratic") == 0) {
    return quadratic;
  } else if (strcmp(name, "catmullrom") == 0) {
    return catmullrom;
  } else if (strcmp(name, "lanczos") == 0) {
    return lanczos<3>;
  } else {
    return nullptr;
  }
}

Magick::FilterTypes parse_kernel_magick(const char* name) {
  if (strcmp(name, "box") == 0) {
    return Magick::FilterTypes::BoxFilter;
  } else if (strcmp(name, "linear") == 0) {
    return Magick::FilterTypes::TriangleFilter;
  } else if (strcmp(name, "quadratic") == 0) {
    return Magick::FilterTypes::QuadraticFilter;
  } else if (strcmp(name, "catmullrom") == 0) {
    return Magick::FilterTypes::CatromFilter;
  } else if (strcmp(name, "lanczos") == 0) {
    return Magick::FilterTypes::LanczosFilter;
  } else {
    return Magick::FilterTypes::PointFilter;
  }
}

template <typename T>
planar_image<T> magick_to_array(const Magick::Image& img) {
  planar_image<T> result({img.columns(), img.rows(), 4});

  const Magick::PixelPacket *pixel_cache = img.getConstPixels(0, 0, result.width(), result.height());
  for (index_t y : result.y()) {
    for (index_t x : result.x()) {
      const Magick::PixelPacket& packet = *(pixel_cache + y*result.width() + x);
      result(x, y, 0) = packet.red;
      result(x, y, 1) = packet.green;
      result(x, y, 2) = packet.blue;
      result(x, y, 3) = packet.opacity;
    }
  }
  return result;
}

template <typename T, typename Shape>
Magick::Image array_to_magick(const array_ref<T, Shape>& img) {
  Magick::Image result(Magick::Geometry(img.width(), img.height()), Magick::Color());
  Magick::PixelPacket *pixel_cache = result.setPixels(0, 0, img.width(), img.height());
  for (index_t y : img.y()) {
    for (index_t x : img.x()) {
      Magick::PixelPacket& packet = *(pixel_cache + y*img.width() + x);
      packet.red = img(x, y, 0);
      packet.green = img(x, y, 1);
      packet.blue = img(x, y, 2);
      packet.opacity = img(x, y, 3);
    }
  }
  result.syncPixels();
  return result;
}

int main(int argc, char* argv[]) {
  Magick::InitializeMagick(*argv);

  if (argc < 6) {
    std::cout << "Usage: " << argv[0]
              << " <input image> <new width> <new height> <kernel> <output image>" << std::endl;
    return -1;
  }

  const char* input_path = argv[1];
  index_t new_width = std::atoi(argv[2]);
  index_t new_height = std::atoi(argv[3]);
  std::function<float(float)> kernel = parse_kernel(argv[4]);
  Magick::FilterTypes magick_kernel = parse_kernel_magick(argv[4]);
  const char* output_path = argv[5];

  Magick::Image image;
  image.read(input_path);

  auto input = magick_to_array<float>(image);

  planar_image<float> output({new_width, new_height, 4});
  const rational<index_t> rate_x(output.width(), input.width());
  const rational<index_t> rate_y(output.height(), input.height());
  double resize_time = benchmark([&]() {
    resize(input.cref(), output.ref(), rate_x, rate_y, kernel);
  });
  std::cout << "resize time: " << resize_time * 1e3 << " ms " << std::endl;

  Magick::Image magick_output = array_to_magick(output.cref());
  magick_output.write(output_path);

  Magick::Image magick_resized(image);

  Magick::Geometry new_size(std::to_string(new_width) + "x" + std::to_string(new_height) + "!");
  magick_resized.filterType(magick_kernel);
  double magick_time = benchmark([&]() {
    magick_resized.resize(new_size);
    image.getPixels(0, 0, new_width, new_height);
  }, true);
  std::cout << "GraphicsMagick time: " << magick_time * 1e3 << " ms " << std::endl;

  planar_image<float> magick_output_array = magick_to_array<float>(magick_output);

  float max_error = 0.0f;
  magick_output_array.for_each_value([&](float i) { max_error = std::max(max_error, i * 1e-2f); });

  for (index_t c : output.c()) {
    for (index_t y : output.y()) {
      for (index_t x : output.x()) {
        if (std::abs(output(x, y, c) - magick_output_array(x, y, c)) > max_error) {
          std::cout << "Image mismatch! " << x << " " << y << " " << c << " " << output(x, y, c) << " " << magick_output_array(x, y, c) << std::endl;
          return -1;
        }
      }
    }
  }
  return 0;
}


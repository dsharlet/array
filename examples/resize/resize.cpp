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
      float k_rx = kernel(rx - in_x);
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
    for (index_t x : out.x()) {
      for (index_t c : out.c()) {
        float out_xyc = 0.0f;
        for (index_t ry : kernel_y.x()) {
          out_xyc += in(x, ry, c) * kernel_y(ry);
        }
        out(x, y, c) = out_xyc;
      }
    }
  }
}

template <typename TIn, typename TOut>
void transpose(const TIn& in, const TOut& out) {
  for (index_t y : out.y()) {
    for (index_t x : out.x()) {
      for (index_t c : out.c()) {
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

  // It's faster to resample in y, then x.
  planar_image<TOut> temp(make_dense(make_shape(in.x(), out.y(), out.c())));
  resize_y(in, temp.ref(), kernels_y);
  planar_image<TOut> temp_tr(make_dense(make_shape(out.y(), in.x(), out.c())));
  //transpose(temp.cref(), temp_tr.ref());
  planar_image<TOut> temp2(make_dense(make_shape(out.y(), out.x(), out.c())));
  resize_y(temp_tr.cref(), temp2.ref(), kernels_x);
  //transpose(temp2.cref(), out.ref());
}

// Define some common kernels useful for resizing images.
float nearest(float x) {
  return -0.5f <= x && x < 0.5f ? 1.0f : 0.0f;
}

float linear(float x) {
  return std::max(0.0f, 1.0f - std::abs(x));
}

float sinc(float x) {
  x *= M_PI;
  return std::abs(x) > 1e-6f ? std::sin(x) / x : 1.0f;
}

float lanczos(float x, int lobes) {
  if (-lobes <= x && x < lobes) {
    return sinc(x) * sinc(x / lobes);
  } else {
    return 0.0f;
  }
}
template<int lobes>
float lanczos(float x) { return lanczos(x, lobes); }


std::function<float(float)> parse_kernel(const char* name) {
  if (strcmp(name, "nearest") == 0) {
    return nearest;
  } else if (strcmp(name, "linear") == 0) {
    return linear;
  } else if (strcmp(name, "lanczos2") == 0) {
    return lanczos<2>;
  } else if (strcmp(name, "lanczos3") == 0) {
    return lanczos<3>;
  } else if (strcmp(name, "lanczos4") == 0) {
    return lanczos<4>;
  } else {
    return nullptr;
  }
}

Magick::FilterTypes parse_kernel_magick(const char* name) {
  if (strcmp(name, "box") == 0) {
    return Magick::FilterTypes::BoxFilter;
  } else if (strcmp(name, "linear") == 0) {
    return Magick::FilterTypes::TriangleFilter;
  } else if (strcmp(name, "lanczos2") == 0) {
    return Magick::FilterTypes::LanczosFilter;
  } else if (strcmp(name, "lanczos3") == 0) {
    return Magick::FilterTypes::LanczosFilter;
  } else if (strcmp(name, "lanczos4") == 0) {
    return Magick::FilterTypes::LanczosFilter;
  } else {
    return Magick::FilterTypes::PointFilter;
  }
}

template <typename T>
dense_array<T, 3> magick_to_array(const Magick::Image& img) {
  dense_array<T, 3> result({img.columns(), img.rows(), 4});

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
  const char* magick_output_path = argc > 6 ? argv[6] : nullptr;

  Magick::Image image;
  image.read(input_path);

  auto input = magick_to_array<float>(image);

  dense_array<float, 3> output({new_width, new_height, 4});
  const rational<index_t> rate_x(output.width(), input.width());
  const rational<index_t> rate_y(output.height(), input.height());
  double resize_time = benchmark([&]() {
    resize(input.cref(), output.ref(), rate_x, rate_y, kernel);
  });
  std::cout << "resize time: " << resize_time * 1e3 << " ms " << std::endl;

  Magick::Image magick_output = array_to_magick(output.cref());
  magick_output.write(output_path);

  if (magick_output_path) {
    Magick::Image magick_resized(image);

    Magick::Geometry new_size(std::to_string(new_width) + "x" + std::to_string(new_height) + "!");
    magick_resized.filterType(magick_kernel);
    double magick_time = benchmark([&]() {
      magick_resized.resize(new_size);
      image.getPixels(0, 0, new_width, new_height);
    }, true);
    std::cout << "GraphicsMagick time: " << magick_time * 1e3 << " ms " << std::endl;

    magick_resized.write(magick_output_path);
  }
  return 0;
}


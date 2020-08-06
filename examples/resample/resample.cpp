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

#include "resample.h"
#include "image.h"

#include <iostream>

#include <Magick++.h>

using namespace nda;

continuous_kernel parse_kernel(const char* name) {
  if (strcmp(name, "box") == 0) {
    return box;
  } else if (strcmp(name, "linear") == 0) {
    return linear;
  } else if (strcmp(name, "quadratic") == 0) {
    return interpolating_quadratic;
  } else if (strcmp(name, "cubic") == 0) {
    return interpolating_cubic;
  } else if (strcmp(name, "lanczos") == 0) {
    return lanczos<4>;
  } else {
    return nullptr;
  }
}

// Need to call img.syncPixels() when done with this.
chunky_image_ref<Magick::Quantum, 4> ref(Magick::Image& img) {
  index_t width = img.columns();
  index_t height = img.rows();
  Magick::PixelPacket* pixel_cache = img.setPixels(0, 0, width, height);
  Magick::Quantum* base = reinterpret_cast<Magick::Quantum*>(pixel_cache);
  return {base, {width, height, 4}};
}

chunky_image_ref<const Magick::Quantum, 4> cref(const Magick::Image& img) {
  index_t width = img.columns();
  index_t height = img.rows();
  const Magick::PixelPacket* pixel_cache = img.getConstPixels(0, 0, width, height);
  const Magick::Quantum* base = reinterpret_cast<const Magick::Quantum*>(pixel_cache);
  return {base, {width, height, 4}};
}

template <class T>
planar_image<T> magick_to_array(const Magick::Image& img) {
  // We can tell make_compact_copy the value_type of the array we want by giving
  // it an allocator for that type.
  auto chunky = cref(img);
  planar_image_shape array_shape(chunky.width(), chunky.height(), chunky.channels());
  return make_copy(chunky, array_shape, std::allocator<T>());
}

template <class T, class Shape>
Magick::Image array_to_magick(const array_ref<T, Shape>& img) {
  Magick::Image result(Magick::Geometry(img.width(), img.height()), Magick::Color());
  copy(img, ref(result));
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
  continuous_kernel kernel = parse_kernel(argv[4]);
  const char* output_path = argv[5];

  Magick::Image image;
  image.read(input_path);

  auto input = magick_to_array<float>(image);

  planar_image<float> output({new_width, new_height, 4});
  const rational<index_t> rate_x(output.width(), input.width());
  const rational<index_t> rate_y(output.height(), input.height());
  resample(input.cref(), output.ref(), rate_x, rate_y, kernel);

  const float max_quantum = std::numeric_limits<Magick::Quantum>::max();
  output.for_each_value([=](float& i) { i = std::max(std::min(i, max_quantum), 0.0f); });

  Magick::Image magick_output = array_to_magick(output.cref());
  magick_output.write(output_path);
  return 0;
}

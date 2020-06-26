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

#include "image.h"
#include "resize.h"

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
  continuous_kernel kernel = parse_kernel(argv[4]);
  const char* output_path = argv[5];

  Magick::Image image;
  image.read(input_path);

  auto input = magick_to_array<float>(image);

  planar_image<float> output({new_width, new_height, 4});
  const rational<index_t> rate_x(output.width(), input.width());
  const rational<index_t> rate_y(output.height(), input.height());
  resize(input.cref(), output.ref(), rate_x, rate_y, kernel);

  Magick::Image magick_output = array_to_magick(output.cref());
  magick_output.write(output_path);
  return 0;
}


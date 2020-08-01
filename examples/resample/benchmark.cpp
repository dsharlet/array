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

#include "benchmark.h"
#include "image.h"
#include "rational.h"
#include "resample.h"

#include <iostream>
#include <utility>

using namespace nda;

const std::pair<const char*, continuous_kernel> benchmarks[] = {
    {"box", box},
    {"linear", linear},
    {"quadratic", interpolating_quadratic},
    {"cubic", interpolating_cubic},
    {"lanczos3", lanczos<3>},
};

template <typename Image, index_t Channels>
void run_benchmarks(
    index_t input_width, index_t input_height, index_t output_width, index_t output_height) {

  Image input({input_width, input_height, Channels});
  Image output({output_width, output_height, Channels});

  const rational<index_t> rate_x(output.width(), input.width());
  const rational<index_t> rate_y(output.height(), input.height());

  for (auto i : benchmarks) {
    double resample_time =
        benchmark([&]() { resample(input.cref(), output.ref(), rate_x, rate_y, i.second); });
    std::cout << i.first << " time: " << resample_time * 1e3 << " ms " << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "Usage: " << argv[0]
              << " <input width> <input height> <output_width> <output height>" << std::endl;
    return -1;
  }

  index_t input_width = std::atoi(argv[1]);
  index_t input_height = std::atoi(argv[2]);
  index_t output_width = std::atoi(argv[3]);
  index_t output_height = std::atoi(argv[4]);

  run_benchmarks<planar_image<float>, 4>(input_width, input_height, output_width, output_height);

  return 0;
}

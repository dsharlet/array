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
#include "benchmark.h"

#include <random>
#include <iostream>

using namespace nda;

template <typename Input, typename Filter, typename Output>
void conv2d(const Input& input, const Filter& filter, const Output& output) {
  for (index_t y : output.y()) {
    for (index_t x : output.x()) {
      for (index_t c : output.c()) {
        output(x, y, c) = 0;
      }
      for (index_t dy : filter.y()) {
        for (index_t dx : filter.x()) {
          for (index_t c : output.c()) {
            output(x, y, c) += input(x + dx, y + dy, c) * filter(dx, dy, c);
          }
        }
      }
    }
  }
}

// TODO: This is slow, fix it.
template <typename Input, typename Filter, typename Output>
void conv2d_tiled(const Input& input, const Filter& filter, const Output& output) {
  using T = typename Output::value_type;

  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);
  constexpr index_t unroll_y = 4;

  for (auto yo : split<unroll_y>(output.y())) {
    for (index_t x : output.x()) {
      for (auto co : split<vector_size>(output.c())) {
        auto output_tile = output(r<1>(x), yo, co);
        T buffer[unroll_y * vector_size] = { static_cast<T>(0) };
        auto accumulator = make_array_ref(buffer, make_compact(output_tile.shape()));
        #pragma unroll
        for (auto y : accumulator.y()) {
          #pragma unroll
          for (index_t dy : filter.y()) {
            #pragma unroll
            for (index_t dx : filter.x()) {
              for (index_t c : accumulator.c()) {
                accumulator(x, y, c) += input(x + dx, y + dy, c) * filter(dx, dy, c);
              }
            }
          }
        }
        for (auto y : output_tile.y()) {
          for (auto c : output_tile.c()) {
            output_tile(x, y, c) = accumulator(x, y, c);
          }
        }
      }
    }
  }
}

int main(int, const char**) {
  constexpr int C = 32;
  constexpr int W = 100;
  constexpr int H = 80;
  constexpr int DX = 3;
  constexpr int DY = 3;

  auto input = make_array<float>(chunky_image_shape<C>(W + DX, H + DY, {}));
  auto filter = make_array<float>(shape<dim<0, DX, C>, dim<0, DY, C*DX>, dense_dim<0, C>>());

  // 'for_each_value' calls the given function with a reference to
  // each value in the array. Use this to randomly initialize the
  // matrices with random values.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  generate(input, [&]() { return uniform(rng); });
  generate(filter, [&]() { return uniform(rng); });

  auto naive_output = make_array<float>(chunky_image_shape<C>(W, H, {}));
  double naive_time = benchmark([&]() {
    conv2d(input.cref(), filter.cref(), naive_output.ref());
  });
  std::cout << "naive time: " << naive_time * 1e3 << " ms" << std::endl;

  auto tiled_output = make_array<float>(chunky_image_shape<C>(W, H, {}));
  double tiled_time = benchmark([&]() {
    conv2d_tiled(input.cref(), filter.cref(), tiled_output.ref());
  });
  std::cout << "tiled time: " << tiled_time * 1e3 << " ms" << std::endl;

  const float epsilon = 1e-4f;
  for_each_index(naive_output.shape(), [&](const index_of_rank<3>& i) {
    if (std::abs(naive_output(i) - tiled_output(i)) > epsilon) {
      std::cout
        << "naive_output(i) = " << naive_output(i)
        << " != tiled_output(i) = " << tiled_output(i) << std::endl;
    }
  });

  return 0;
}


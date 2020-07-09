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
#include "benchmark.h"

#include <random>
#include <iostream>

using namespace nda;

template <typename Input, typename Filter, typename Output>
void conv_naive(const Input& input, const Filter& filter, const Output& output) {
  for (index_t n : output.template dim<3>()) {
    for (index_t y : output.template dim<2>()) {
      for (index_t x : output.template dim<1>()) {
        for (index_t co : output.template dim<0>()) {
          output(co, x, y, n) = 0;
          for (index_t ci : filter.template dim<3>()) {
            for (index_t dy : filter.template dim<2>()) {
              for (index_t dx : filter.template dim<1>()) {
                output(co, x, y, n) += filter(co, dx, dy, ci) * input(ci, x + dx, y + dy, n);
              }
            }
          }
        }
      }
    }
  }
}

template <typename Input, typename Filter, typename Output>
__attribute__((always_inline))
void conv(const Input& input, const Filter& filter, const Output& output) {
  typedef typename Output::value_type T;

  for (index_t n : output.template dim<3>()) {
    for (index_t y : output.template dim<2>()) {
#if 1
      fill(output(_, _, y, n), static_cast<T>(0));
#else
      for (index_t x : output.template dim<1>()) {
        for (index_t co : output.template dim<0>()) {
          output(co, x, y, n) = 0;
        }
      }
#endif
      for (index_t ci : filter.template dim<3>()) {
        for (index_t dy : filter.template dim<2>()) {
          for (index_t dx : filter.template dim<1>()) {
            for (index_t x : output.template dim<1>()) {
              for (index_t co : output.template dim<0>()) {
                output(co, x, y, n) += filter(co, dx, dy, ci) * input(ci, x + dx, y + dy, n);
              }
            }
          }
        }
      }
    }
  }
}

template <typename Input, typename Filter, typename Output>
void conv_tiled(const Input& input, const Filter& filter, const Output& output) {
  typedef typename Output::value_type T;

  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);

  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr index_t tile_x = 4;
  constexpr index_t tile_co = vector_size * 3;

  for (index_t n : output.template dim<3>()) {
    for (index_t y : output.template dim<2>()) {
      for (auto xo : split<tile_x>(output.template dim<1>())) {
        for (auto coo : split<tile_co>(output.template dim<0>())) {
          // Don't slice the y, n.template dims by making them a range here.
          auto output_tile = output(coo, xo, fixed_range<1>(y), fixed_range<1>(n));
#if 1
          // TODO: This is slow, probably due to potential aliasing that
          // we can't fix due to https://bugs.llvm.org/show_bug.cgi?id=45863
          conv(input, filter, output_tile);
#elif 0
          // TODO: This is slow, probably due to potential aliasing that
          // we can't fix due to https://bugs.llvm.org/show_bug.cgi?id=45863
          T buffer[tile_x * tile_co] = { 0 };
          auto accumulator = make_array_ref(buffer, make_compact(output_tile.shape()));
          conv(input, filter, accumulator);
          for (index_t x : xo) {
            for (index_t co : coo) {
              output(co, x, y, n) = accumulator(co, x, y, n);
            }
          }
#else
          // TODO: This scalarizes for some unknown reason.
          T buffer[tile_x * tile_co] = { 0 };
          auto accumulator = make_array_ref(buffer, make_compact(output_tile.shape()));
#  if 1
#    if 0
          fill(accumulator, static_cast<T>(0));
#    else
          for (index_t x : xo) {
            for (index_t co : coo) {
              accumulator(co, x, y, n) = 0;
            }
          }
#    endif
#  endif
          for (index_t ci : filter.template dim<3>()) {
            for (index_t dy : filter.template dim<2>()) {
              for (index_t dx : filter.template dim<1>()) {
                for (index_t x : xo) {
                  for (index_t co : coo) {
                    accumulator(co, x, y, n) += filter(co, dx, dy, ci) * input(ci, x + dx, y + dy, n);
                  }
                }
              }
            }
          }
#  if 0
          copy(accumulator, output_tile);
#  else
          for (index_t x : xo) {
            for (index_t co : coo) {
              output(co, x, y, n) = accumulator(co, x, y, n);
            }
          }
#  endif
#endif
        }
      }
    }
  }
}

// Define a fully compile-time constant shape.
template <index_t X, index_t Y, index_t Z, index_t W>
using tensor_shape = shape<
    dense_dim<0, X>,
    dim<0, Y, X>,
    dim<0, Z, X * Y>,
    dim<0, W, X * Y * Z>>;

int main(int, const char**) {
  constexpr int N = 5;
  constexpr int CI = 128;
  constexpr int CO = 128;
  constexpr int W = 100;
  constexpr int H = 80;

  auto input = make_array<float>(tensor_shape<CI, W + 2, H + 2, N>());
  auto filter = make_array<float>(tensor_shape<CO, 3, 3, CI>());

  // 'for_each_value' calls the given function with a reference to
  // each value in the array. Use this to randomly initialize the
  // matrices with random values.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  generate(input, [&]() { return uniform(rng); });
  generate(filter, [&]() { return uniform(rng); });

  auto naive_output = make_array<float>(tensor_shape<CO, W, H, N>());
  double naive_time = benchmark([&]() {
    conv_naive(input.cref(), filter.cref(), naive_output.ref());
  });
  std::cout << "naive time: " << naive_time * 1e3 << " ms" << std::endl;

  auto tiled_output = make_array<float>(tensor_shape<CO, W, H, N>());
  double tiled_time = benchmark([&]() {
    conv_tiled(input.cref(), filter.cref(), tiled_output.ref());
  });
  std::cout << "tiled time: " << tiled_time * 1e3 << " ms" << std::endl;

  const float epsilon = 1e-4f;
  for_each_index(naive_output.shape(), [&](const shape_of_rank<4>::index_type& i) {
    if (std::abs(naive_output(i) - tiled_output(i)) > epsilon) {
      std::cout
        << "naive_output(i) = " << naive_output(i)
        << " != tiled_output(i) = " << tiled_output(i) << std::endl;
    }
  });

  return 0;
}


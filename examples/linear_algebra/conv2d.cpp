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
        for (index_t dy : filter.y()) {
          for (index_t dx : filter.x()) {
            output(x, y, c) += input(x + dx, y + dy, c) * filter(dx, dy, c);
          }
        }
      }
    }
  }
}

// TODO: This is slow, fix it. The C version below is fast :(
template <typename Input, typename Filter, typename Output>
void conv2d_tiled(const Input& input, const Filter& filter, const Output& output) {
  using T = typename Output::value_type;

  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);

  // The strategy used here is:
  // - vectorize the channel dimension (assume it has stride 1)
  // - tile x and y to get some re-use of the stencil, and to
  //   get ILP for the accumulators.
  constexpr index_t unroll_x = 2;
  constexpr index_t unroll_y = 4;

  for (auto yo : split<unroll_y>(output.y())) {
    for (auto xo : split<unroll_x>(output.x())) {
      for (auto co : split<vector_size>(output.c())) {
        auto output_tile = output(xo, yo, co);
        T buffer[unroll_x * unroll_y * vector_size] = { static_cast<T>(0) };
        auto accumulator = make_array_ref(buffer, make_compact(output_tile.shape()));
        for (index_t dy : filter.y()) {
          for (index_t dx : filter.x()) {
            #pragma unroll
            for (index_t y : accumulator.y()) {
              #pragma unroll
              for (index_t x : accumulator.x()) {
                for (index_t c : accumulator.c()) {
                  accumulator(x, y, c) += input(x + dx, y + dy, c) * filter(dx, dy, c);
                }
              }
            }
          }
        }
        for (index_t y : output_tile.y()) {
          for (index_t x : output_tile.x()) {
            for (index_t c : output_tile.c()) {
              output_tile(x, y, c) = accumulator(x, y, c);
            }
          }
        }
      }
    }
  }
}


// This generates the following inner loop:
// LBB12_6:
//   vmovups -256(%r8,%r14), %ymm9
//   vfmadd231ps     -384(%rdi,%r15), %ymm9, %ymm1
//   vmovups -256(%rdi,%r15), %ymm10
//   vfmadd231ps     %ymm10, %ymm9, %ymm2
//   vfmadd231ps     -384(%r12,%r15), %ymm9, %ymm3
//   vmovups -256(%r12,%r15), %ymm11
//   vfmadd231ps     -384(%r10,%r15), %ymm9, %ymm5
//   vfmadd231ps     %ymm11, %ymm9, %ymm4
//   vmovups -256(%r10,%r15), %ymm12
//   vfmadd231ps     %ymm12, %ymm9, %ymm6
//   vfmadd231ps     -384(%r13,%r15), %ymm9, %ymm7
//   vmovups -256(%r13,%r15), %ymm13
//   vfmadd213ps     %ymm8, %ymm13, %ymm9
//   vmovups -128(%r8,%r14), %ymm8
//   vfmadd231ps     %ymm10, %ymm8, %ymm1
//   vmovups -128(%rdi,%r15), %ymm10
//   vfmadd231ps     %ymm11, %ymm8, %ymm3
//   vmovups -128(%r12,%r15), %ymm11
//   vfmadd231ps     %ymm12, %ymm8, %ymm5
//   vmovups -128(%r10,%r15), %ymm12
//   vfmadd231ps     %ymm13, %ymm8, %ymm7
//   vmovups -128(%r13,%r15), %ymm13
//   vfmadd231ps     %ymm10, %ymm8, %ymm2
//   vfmadd231ps     %ymm11, %ymm8, %ymm4
//   vfmadd231ps     %ymm12, %ymm8, %ymm6
//   vfmadd213ps     %ymm9, %ymm13, %ymm8
//   vmovups (%r8,%r14), %ymm9
//   vfmadd231ps     %ymm10, %ymm9, %ymm1
//   vfmadd231ps     %ymm11, %ymm9, %ymm3
//   vfmadd231ps     %ymm12, %ymm9, %ymm5
//   vfmadd231ps     (%rdi,%r15), %ymm9, %ymm2
//   vfmadd231ps     (%r12,%r15), %ymm9, %ymm4
//   vfmadd231ps     (%r10,%r15), %ymm9, %ymm6
//   vfmadd231ps     %ymm13, %ymm9, %ymm7
//   vfmadd231ps     (%r13,%r15), %ymm9, %ymm8
//   addq    $384, %r14
//   addq    %rsi, %r15
//   cmpq    $1408, %r14
//   jne     LBB12_6

template <index_t Channels, index_t DX, index_t DY>
void conv2d_tiled_c(
    const float* input, const float* filter, float* output,
    index_t width, index_t height, index_t input_stride, index_t output_stride) {
  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(float);

  // The strategy used here is:
  // - vectorize the channel dimension (assume it has stride 1)
  // - tile x and y to get some re-use of the stencil, and to
  //   get ILP for the accumulators.
  constexpr index_t unroll_x = 2;
  constexpr index_t unroll_y = 4;

  for (index_t yo = 0; yo < height; yo += unroll_y) {
    for (index_t xo = 0; xo < width; xo += unroll_x) {
      // TODO: Putting this outermost should be better, but it's slower.
      for (index_t co = 0; co < Channels; co += vector_size) {
        float buffer[unroll_x * unroll_y * vector_size] = { 0.0f };
        for (index_t dy = 0; dy < DY; dy++) {
          for (index_t dx = 0; dx < DX; dx++) {
            #pragma unroll
            for (index_t yi = 0; yi < unroll_y; yi++) {
              #pragma unroll
              for (index_t xi = 0; xi < unroll_x; xi++) {
                for (index_t ci = 0; ci < vector_size; ci++) {
                  index_t x = xo + xi;
                  index_t y = yo + yi;
                  index_t c = co + ci;
                  buffer[xi * vector_size + yi * vector_size * unroll_x + ci] +=
                      input[(x + dx) * Channels + (y + dy) * input_stride + c] * filter[dy * DX * Channels + dx * Channels + c];
                }
              }
            }
          }
        }
        for (index_t yi = 0; yi < unroll_y; yi++) {
          for (index_t xi = 0; xi < unroll_x; xi++) {
            for (index_t ci = 0; ci < vector_size; ci++) {
              index_t x = xo + xi;
              index_t y = yo + yi;
              index_t c = co + ci;
              output[x * Channels + y * output_stride + c] = buffer[xi * vector_size + yi * vector_size * unroll_x + ci];
            }
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

  auto input = make_array<float>(chunky_image_shape<C, nda::dynamic>(W + DX, H + DY, {}));
  auto filter = make_array<float>(shape<dim<0, DX, C>, dim<0, DY, C*DX>, dense_dim<0, C>>());

  // 'for_each_value' calls the given function with a reference to
  // each value in the array. Use this to randomly initialize the
  // matrices with random values.
  std::mt19937_64 rng;
  std::uniform_real_distribution<float> uniform(0, 1);
  generate(input, [&]() { return uniform(rng); });
  generate(filter, [&]() { return uniform(rng); });

  auto naive_output = make_array<float>(chunky_image_shape<C, nda::dynamic>(W, H, {}));
  double naive_time = benchmark([&]() {
    conv2d(input.cref(), filter.cref(), naive_output.ref());
  });
  std::cout << "naive time: " << naive_time * 1e3 << " ms" << std::endl;

  auto tiled_output = make_array<float>(chunky_image_shape<C, nda::dynamic>(W, H, {}));
  double tiled_time = benchmark([&]() {
    conv2d_tiled(input.cref(), filter.cref(), tiled_output.ref());
  });
  std::cout << "tiled time: " << tiled_time * 1e3 << " ms" << std::endl;

  auto tiled_output_c = make_array<float>(chunky_image_shape<C, nda::dynamic>(W, H, {}));
  double tiled_time_c = benchmark([&]() {
    conv2d_tiled_c<C, DX, DY>(
        input.data(), filter.data(), tiled_output_c.data(),
        tiled_output_c.width(), tiled_output_c.height(),
        input.y().stride(), tiled_output_c.y().stride());
  });
  std::cout << "tiled time (C): " << tiled_time_c * 1e3 << " ms" << std::endl;

  const float epsilon = 1e-4f;
  for_each_index(naive_output.shape(), [&](const index_of_rank<3>& i) {
    if (std::abs(naive_output(i) - tiled_output(i)) > epsilon) {
      std::cout
        << "naive_output(i) = " << naive_output(i)
        << " != tiled_output(i) = " << tiled_output(i) << std::endl;
    }
    if (std::abs(naive_output(i) - tiled_output_c(i)) > epsilon) {
      std::cout
        << "naive_output(i) = " << naive_output(i)
        << " != tiled_output_c(i) = " << tiled_output_c(i) << std::endl;
    }
  });

  return 0;
}


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
#include "test.h"

namespace nda {

template <typename Shape>
void test_crop() {
  array<int, Shape> base({100, 80, 3});
  fill_pattern(base);

  auto crop_xy = crop(base, 2, 1, 96, 77, crop_origin::crop);
  check_pattern(crop_xy);

  auto crop_zero_xy = crop(base, 3, 2, 95, 76, crop_origin::zero);
  check_pattern(crop_zero_xy, std::make_tuple(3, 2, 0));

  auto crop2_xy = crop(crop_xy, 4, 3, 92, 73, crop_origin::crop);
  check_pattern(crop2_xy);
}

TEST(image_crop) {
  test_crop<planar_image_shape>();
  test_crop<chunky_image_shape<3>>();
}

template <typename Shape>
void test_slice(index_t channels) {
  array<int, Shape> base({100, 80, channels});
  fill_pattern(base);

  for (index_t c = 0; c < channels; c++) {
    auto slice = slice_channel(base, c);
    for_all_indices(slice.shape(), [&](int x, int y) {
      ASSERT_EQ(slice(x, y), pattern<int>(std::make_tuple(x, y, c)));
    });

    auto crop_slice = slice_channel(crop(base, 5, 3, 80, 70, crop_origin::crop), c);
    for_all_indices(crop_slice.shape(), [&](int x, int y) {
      ASSERT_EQ(crop_slice(x, y), pattern<int>(std::make_tuple(x, y, c)));
    });
  }
}

TEST(image_slice) {
  test_slice<planar_image_shape>(5);
  test_slice<chunky_image_shape<5>>(5);
}

template <typename T, typename ShapeSrc, typename ShapeDest>
void test_copy(index_t channels) {
  array<T, ShapeSrc> src({40, 30, channels});
  fill_pattern(src);

  array<T, ShapeDest> dest({src.width(), src.height(), channels});
  copy(src, dest);
  check_pattern(dest);

  array<T, ShapeDest> dest_cropped({{5, 30}, {3, 20}, channels});
  copy(src, dest);
  check_pattern(dest);

  // If the src and dest shapes are the same, we should expect copies
  // to be about as fast as memcpy, even if the dest is cropped (so
  // some cleverness is required to copy rows at a time).
  // TODO: It would be nice if this were fast even for different shapes.
  // This may be impossible on x86. On ARM, using vstN/vldN, it may
  // be possible, but we need to find a way to sort dimensions by stride
  // while preserving the compile time constant extent.
#if 0
  if (std::is_same<ShapeSrc, ShapeDest>::value) {
    double copy_time = benchmark([&]() {
      copy(src, dest_cropped);
    });
    check_pattern(dest_cropped);

    array<T, ShapeDest> dest_memcpy(dest_cropped.shape());
    double memcpy_time = benchmark([&]() {
      std::memcpy(dest_memcpy.base(), src.base(), dest_cropped.size() * sizeof(T));
    });
    // This memcpy is *not* correct, but the performance of it
    // is optimistic.
    ASSERT(dest_memcpy != dest_cropped);

    ASSERT_LT(copy_time, memcpy_time * 1.2);
  }
#endif
}

template <typename ShapeSrc, typename ShapeDest>
void test_copy_all_types(index_t channels) {
  test_copy<int32_t, ShapeSrc, ShapeDest>(channels);
  test_copy<int16_t, ShapeSrc, ShapeDest>(channels);
  test_copy<int8_t, ShapeSrc, ShapeDest>(channels);
}

TEST(image_chunky_copy) {
  test_copy_all_types<chunky_image_shape<1>, chunky_image_shape<1>>(1);
  test_copy_all_types<chunky_image_shape<2>, chunky_image_shape<2>>(2);
  test_copy_all_types<chunky_image_shape<3>, chunky_image_shape<3>>(3);
  test_copy_all_types<chunky_image_shape<4>, chunky_image_shape<4>>(4);
}

TEST(image_planar_copy) {
  for (int i = 1; i <= 4; i++) {
    test_copy_all_types<planar_image_shape, planar_image_shape>(i);
  }
}

// Copying from planar to chunky is an "interleaving" operation.
TEST(image_interleave) {
  test_copy_all_types<planar_image_shape, chunky_image_shape<1>>(1);
  test_copy_all_types<planar_image_shape, chunky_image_shape<2>>(2);
  test_copy_all_types<planar_image_shape, chunky_image_shape<3>>(3);
  test_copy_all_types<planar_image_shape, chunky_image_shape<4>>(4);
}

// Copying from chunky to planar is a "deinterleaving" operation.
TEST(image_deinterleave) {
  test_copy_all_types<chunky_image_shape<1>, planar_image_shape>(1);
  test_copy_all_types<chunky_image_shape<2>, planar_image_shape>(2);
  test_copy_all_types<chunky_image_shape<3>, planar_image_shape>(3);
  test_copy_all_types<chunky_image_shape<4>, planar_image_shape>(4);
}

TEST(image_chunky_padded) {
  chunky_image<int, 4> src({40, 30, 4});
  fill_pattern(src);
  chunky_image<int, 4> dest(src.shape(), 5);

  chunky_image_ref<const int, 3, 4> src_rgb(src.base(), {src.width(), src.height(), 3});
  chunky_image_ref<int, 3, 4> dest_rgb(dest.base(), {src.width(), src.height(), 3});
  copy(src_rgb, dest_rgb);

  for (int y = 0; y < dest.height(); y++) {
    for (int x = 0; x < dest.width(); x++) {
      ASSERT_EQ(dest(x, y, 0), src(x, y, 0));
      ASSERT_EQ(dest(x, y, 1), src(x, y, 1));
      ASSERT_EQ(dest(x, y, 2), src(x, y, 2));
      // The channel corresponding to padding should not have been
      // overwritten by the pattern.
      ASSERT_EQ(dest(x, y, 3), 5);
    }
  }
}

}  // namespace nda

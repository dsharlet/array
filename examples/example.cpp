#include "array.h"
#include "benchmark.h"

#include <iostream>
#ifdef __AVX2__
#include <immintrin.h>
#endif

using namespace array;

int main(int argc, const char** argv) {
  // Define a shape type for our arrays. Images are often represented
  // by 3 dimensional arrays (width, height, channels). Each dimension
  // is represented by a 'dim' object.
  typedef shape<dim<>, dim<>, dim<>> image_shape_type;

  // We can use this to declare a particular image shape:
  constexpr int width = 4000;
  constexpr int height = 3000;
  constexpr int channels = 3;
  image_shape_type image_shape(width, height, channels);

  // And we can now use this shape to declare a specific array:
  typedef array<int, image_shape_type> image_type;
  image_type image(image_shape);

  // We can also declare arrays implicitly constructing the
  // shape:
  image_type other_image({width, height, channels});

  // We can initialize the array with a normal set of loops
  // using array's operator():
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        image(x, y, c) = x + y;
      }
    }
  }

  // We can also use 'for_all_indices' to traverse the array shape:
  for_all_indices(other_image.shape(), [&](int x, int y, int c) {
    other_image(x, y, c) = x * y;
  });

  // Or we can use 'for_each_index':
  image_type sum_image({width, height, channels});
  double sum_benchmark = benchmark([&]() {
    // TODO: This is currently as fast as the dense code below. This
    // is probably because the compiler is "cheating" to learn that
    // the stride of x is one by inlining and constant folding
    // everything. This won't always be possible.
    for_each_index(sum_image.shape(), [&](const image_type::index_type& index) {
      sum_image(index) = image(index) + other_image(index);
    });
  });
  std::cout << "Image sum took " << sum_benchmark * 1e3 << " ms" << std::endl;

  // The above images we've been working with so far are getting the
  // default memory layout for array shapes, which is the innermost
  // dimension 'width' getting the smallest stride 1, and increasing
  // from there.  In this layout, the 'channels' are stored
  // separately, often called a 'planar' image.
  //
  // A problem with what we've done so far is that the compiler doesn't
  // know that the stride of the width dimension is 1 at compile time.
  // This means that the compiler will do extra work to compute the
  // addresses of each pixel, and will not be able to vectorize code
  // like this.
  //
  // To avoid this, we can use a 'dense_dim' object instead of a 'dim'
  // object to describe that stride 1 dimension. 'dim' objects are
  // templates, which take constant 'min', 'extent', and 'stride'
  // template parameters. These default to 'UNKNOWN', which means the
  // runtime value of this parameter is used instead of the compile
  // time parameter. 'dense_dim' is a synonym for
  // 'dim<UNKNOWN, UNKNOWN, 1>'.
  typedef array<int, shape<dense_dim<>, dim<>, dim<>>> planar_image_type;
  planar_image_type planar_image({width, height, channels});
  planar_image_type planar_other_image({width, height, channels});

  // While array does have a full set of copy constructors and
  // assignment operators, these images have different types because
  // their shape types are different, so we need to use the 'copy'
  // helper function to copy to an image with a different type of
  // shape.
  copy(image, planar_image);
  copy(other_image, planar_other_image);

  // We can use the same code above to process these images:
  planar_image_type planar_sum_image({width, height, channels});
  double planar_sum_benchmark = benchmark([&]() {
    for_each_index(planar_sum_image.shape(), [&](const image_type::index_type& index) {
      planar_sum_image(index) =
          planar_image(index) + planar_other_image(index);
    });
  });
  std::cout << "Planar image sum took "
            << planar_sum_benchmark * 1e3 << " ms" << std::endl;

  // Arrays with a dense first dimension are common enough that these
  // have a built in alias 'dense_shape' and 'dense_array'.

#ifdef __AVX2__
  // For comparison, here is the code above implemented with SIMD
  // intrinsics.
  planar_image_type planar_sum_intrinsics_image({width, height, channels});
  double planar_sum_intrinsics_benchmark = benchmark([&]() {
    for (int y = 0; y < height; y++) {
      for (int c = 0; c < channels; c++) {
        const int* image_row = &planar_image(0, y, c);
        const int* other_image_row = &planar_other_image(0, y, c);
        int* sum_intrinsics_row = &planar_sum_intrinsics_image(0, y, c);
        int x = 0;
        for (; x + 7 < width; x += 8) {
          __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&image_row[x]));
          __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other_image_row[x]));
          __m256i c = _mm256_add_epi32(a, b);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(&sum_intrinsics_row[x]), c);
        }
        for (; x < width; x++) {
          sum_intrinsics_row[x] = image_row[x] + other_image_row[x];
        }
      }
    }
  });
  std::cout << "Planar image sum (using intrinsics) took "
            << planar_sum_intrinsics_benchmark * 1e3 << " ms" << std::endl;
#endif

  // Another common form of image layout is an 'interleaved' image,
  // where the dense dimension is the channel dimension. For this
  // layout, it is also very helpful for the compiler to understand
  // that the number of channels, and the stride of the width
  // dimension is constant, in addition to the dense dimension. It's
  // also often necessary to pad these images to 4 channels. We can
  // express this with custom 'dim<>' objects.
  typedef shape<dim<0, UNKNOWN, 4>, dim<>, dense_dim<0, channels>> interleaved_shape_type;
  typedef array<int, interleaved_shape_type> interleaved_image_type;
  interleaved_image_type interleaved_image({width, height, channels});
  interleaved_image_type interleaved_other_image({width, height, channels});
  interleaved_image_type interleaved_sum_image({width, height, channels});
  copy(image, interleaved_image);
  copy(other_image, interleaved_other_image);

  // In this case, the default loop traversal order (innermost to
  // outermost) is not good due to poor locality. We can avoid this by
  // writing our own loops.
  double interleaved_sum_benchmark = benchmark([&]() {
    // TODO: This is currently much slower than either planar image sum.
    for (int y : interleaved_sum_image.shape().template dim<1>()) {
      for (int x : interleaved_sum_image.shape().template dim<0>()) {
        for (int c : interleaved_sum_image.shape().template dim<2>()) {
          interleaved_sum_image(x, y, c) =
              interleaved_image(x, y, c) + interleaved_other_image(x, y, c);
        }
      }
    };
  });
  std::cout << "Interleaved image sum took "
            << interleaved_sum_benchmark * 1e3 << " ms" << std::endl;
  
  // Check that the results of each computation above are correct.
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        int expected = x + y + x * y;
        if (sum_image(x, y, c) != expected) {
          std::cout << "Error! sum_image(" << x << ", " << y << ", " << c << ") = "
                    << sum_image(x, y, c) << " != " << expected << std::endl;
          abort();
        }
        if (planar_sum_image(x, y, c) != expected) {
          std::cout << "Error! planar_sum_image(" << x << ", " << y << ", " << c << ") = "
                    << planar_sum_image(x, y, c) << " != " << expected << std::endl;
          abort();
        }
        if (interleaved_sum_image(x, y, c) != expected) {
          std::cout << "Error! interleaved_sum_image(" << x << ", " << y << ", " << c << ") = "
                    << interleaved_sum_image(x, y, c) << " != " << expected << std::endl;
          abort();
        }
      }
    }
  }

  return 0;
}

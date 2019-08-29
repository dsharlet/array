#include "array.h"
#include "benchmark.h"

#include <iostream>

using namespace array;

// Blur 5 integer samples with a 1 4 6 4 1 kernel.
template <typename T>
T gaussian_blur_5(T x0, T x1, T x2, T x3, T x4) {
  return (x0 + x1*4 + x2*6 + x3*4 + x4 + 8) / 16;
}

// 2D blur an image with a 1 4 6 4 1 kernel in each dimension.
template <typename Image>
void gaussian_blur_5x5(const Image& in, Image& out) {
  auto x_dim = out.template dim<0>();
  auto y_dim = out.template dim<1>();
  auto c_dim = out.template dim<2>();
  // The strategy here is to blur each row in y first, then blur that
  // in x, and do this per line of the output.
  // This means we need to store one line of the blur in y, so let's
  // define a shape of a buffer for a single line.
  auto line_buffer_shape = make_shape(x_dim, c_dim);
  auto blur_y = make_array<typename Image::value_type>(line_buffer_shape);
  for (int y : y_dim) {
    // Blur in y first.
    for (int x : x_dim) {
      for (int c : c_dim) {
        auto y0 = in(x, clamp(y - 2, y_dim), c);
        auto y1 = in(x, clamp(y - 1, y_dim), c);
        auto y2 = in(x, clamp(y, y_dim), c);
        auto y3 = in(x, clamp(y + 1, y_dim), c);
        auto y4 = in(x, clamp(y + 2, y_dim), c);
        blur_y(x, c) = gaussian_blur_5(y0, y1, y2, y3, y4);
      }
    }

    // Blur in x.
    for (int x : x_dim) {
      for (int c : c_dim) {
        auto x0 = blur_y(clamp(x - 2, x_dim), c);
        auto x1 = blur_y(clamp(x - 1, x_dim), c);
        auto x2 = blur_y(clamp(x, x_dim), c);
        auto x3 = blur_y(clamp(x + 1, x_dim), c);
        auto x4 = blur_y(clamp(x + 2, x_dim), c);
        out(x, y, c) = gaussian_blur_5(x0, x1, x2, x3, x4);
      }
    }
  }
}

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
  typedef array<uint8_t, image_shape_type> image_type;
  image_type image(image_shape);

  // 'for_all_indices' calls a function with each possible value in
  // the array. We can use this to initialize the contents of the
  // array using 'array::operator()'.
  for_all_indices(image.shape(), [&](int x, int y, int c) {
    image(x, y, c) = static_cast<uint8_t>(x + y);
  });

  // We can also declare arrays directly, without declaring a shape
  // first:
  image_type blurred({width, height, channels});
  std::cout << "gaussian_blur_5x5(image_type)... ";
  double time = benchmark([&]() {
    gaussian_blur_5x5(image, blurred);
  });
  std::cout << time * 1e3 << " ms" << std::endl;

  // The above images we've been working with so far are getting the
  // default memory layout for array shapes, which is the innermost
  // dimension 'width' getting the smallest stride 1, and increasing
  // from there.  In this layout, the 'channels' are stored
  // separately, often called a 'planar' image.
  //
  // 'image_shape' is very flexible; it allows any strides. However,
  // this flexibility is a potential performance problem, because the
  // compiler doesn't know that the stride of the width dimension is 1
  // at compile time.  This means that the compiler will do extra work
  // to compute the addresses of each pixel, and will not be able to
  // vectorize code like this.
  //
  // To avoid this, we can use a 'dense_dim' object instead of a 'dim'
  // object to describe that stride 1 dimension, and we can use the
  // template parameters of 'dim' objects to describe known constant
  // 'min', 'extent', and 'stride' parameters. These default to
  // 'UNK', which means the runtime value of this parameter is
  // used instead of the compile time parameter. 'dense_dim' is a
  // synonym for 'dim<UNK, UNK, 1>'.
  typedef array<uint8_t, shape<dense_dim<>, dim<>, dim<0, channels>>> planar_image_type;
  planar_image_type planar_image({width, height, channels});

  // While array does have a full set of copy constructors and
  // assignment operators, these images have different types because
  // their shape types are different, so we need to use the 'copy'
  // helper function to copy to an image with a different type of
  // shape.
  copy(image, planar_image);

  // We can use the same code above to process these images:
  planar_image_type planar_blurred({width, height, channels});
  std::cout << "gaussian_blur_5x5(planar_image_type)... ";
  double planar_time = benchmark([&]() {
    gaussian_blur_5x5(planar_image, planar_blurred);
  });
  std::cout << planar_time * 1e3 << " ms" << std::endl;

  // Another common form of image layout is an 'interleaved' image,
  // where the dense dimension is the channel dimension. For this
  // layout, it is also very helpful for the compiler to understand
  // that the number of channels, and the stride of the width
  // dimension is constant, in addition to the dense dimension. It's
  // also often necessary to pad these images to 4 channels (32 bits
  // per pixel). We can express this with custom 'dim<>' objects.
  typedef shape<dim<UNK, UNK, 4>, dim<>, dense_dim<0, channels>> interleaved_shape_type;
  typedef array<uint8_t, interleaved_shape_type> interleaved_image_type;
  interleaved_image_type interleaved_image({width, height, channels});
  copy(image, interleaved_image);

  // In this case, the default loop traversal order (innermost to
  // outermost) is not good due to poor locality. We can avoid this by
  // writing our own loops.
  interleaved_image_type interleaved_blurred({width, height, channels});
  std::cout << "gaussian_blur_5x5(interleaved_image_type)... ";
  double interleaved_time = benchmark([&]() {
    gaussian_blur_5x5(interleaved_image, interleaved_blurred);
  });
  std::cout << interleaved_time * 1e3 << " ms" << std::endl;
  
  // Check that the results of each computation above are correct.
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        const int kernel[5] = { 1, 4, 6, 4, 1 };
        int blur_x_sum = 0;
        for (int dx = -2; dx <= 2; dx++) {
          int blur_y_sum = 0;
          for (int dy = -2; dy <= 2; dy++) {
            int tap =
              image(clamp(x + dx, 0, width - 1), clamp(y + dy, 0, height - 1), c);
            blur_y_sum += tap * kernel[dy + 2];
          }
          blur_x_sum += ((blur_y_sum + 8) / 16) * kernel[dx + 2];
        }
        int expected = (blur_x_sum + 8) / 16;

        if (blurred(x, y, c) != expected) {
          std::cout << "Error! blurred(" << x << ", " << y << ", " << c << ") = "
                    << blurred(x, y, c) << " != " << expected << std::endl;
          abort();
        }
        if (planar_blurred(x, y, c) != expected) {
          std::cout << "Error! planar_blurred(" << x << ", " << y << ", " << c << ") = "
                    << planar_blurred(x, y, c) << " != " << expected << std::endl;
          abort();
        }
        if (interleaved_blurred(x, y, c) != expected) {
          std::cout << "Error! interleaved_blurred(" << x << ", " << y << ", " << c << ") = "
                    << interleaved_blurred(x, y, c) << " != " << expected << std::endl;
          abort();
        }
      }
    }
  }

  return 0;
}

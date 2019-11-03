#include "array.h"
#include "test.h"
#include "performance.h"

namespace array {

typedef shape<strided_dim<3>, dim<>, dense_dim<0, 3>> interleaved_shape;

TEST(performance_dense3d_assignment) {
  const int width = 1024;
  const int height = 1024;
  const int depth = 50;

  dense_array<int, 3> a({width, height, depth});
  double loop_time = time_ms([=, &a] {
    for (int z = 0; z < depth; z++) {
      for (int y = 0; y < height; y++) {
	int *row_data = &a(0, y, z);
	for (int x = 0; x < width; x++) {
	  row_data[x] = z * width * height + y * width + x;
	}
      }
    }
  });
  assert_used(a);

  // Set every value using for_all_indices.
  dense_array<int, 3> b({width, height, depth});
  double for_all_indices_time = time_ms([=, &b]() {
    for_all_indices(b.shape(), [&](int x, int y, int z) {
      b(x, y, z) = z * width * height + y * width + x;
    });
  });
  assert_used(b);

  // Set every value using for_each_index.
  dense_array<int, 3> c({width, height, depth});
  double for_each_index_time = time_ms([=, &c]() {
    for_each_index(c.shape(), [&](const dense_array<int, 3>::index_type& i) {
      c(i) = std::get<2>(i) * width * height + std::get<1>(i) * width + std::get<0>(i);
    });
  });
  assert_used(c);

  // Check that timings are acceptable. These are actually a lot faster!?
  const double tolerance = loop_time * 0.2f;
  ASSERT_REQ(for_all_indices_time, loop_time * 0.66f, tolerance);
  ASSERT_REQ(for_each_index_time, loop_time * 0.66f, tolerance);
}

TEST(performance_dense3d_copy) {
  int width = 500;
  int height = 200;
  int depth = 100;

  dense_array<int, 3> a({width, height, depth}, 3);

  dense_array<int, 3> b({width, height, depth});
  double copy_time = time_ms([&]() {
    copy(a, b);
  });
  assert_used(b);

  dense_array<int, 3> c({width, height, depth});
  ASSERT_EQ(c.shape().flat_extent(), c.size());
  double memcpy_time = time_ms([&] {
    memcpy(&c(0, 0, 0), &a(0, 0, 0), a.size() * sizeof(int));
  });
  assert_used(c);

  // copy should be about as fast as memcpy.
  ASSERT_REQ(copy_time, memcpy_time, memcpy_time * 0.2f);
}

TEST(performance_interleaved_copy) {
  int width = 5000;
  int height = 500;
  int depth = 3;

  array<int, interleaved_shape> a({width, height, depth}, 3);

  array<int, interleaved_shape> b({width, height, depth});
  double copy_time = time_ms([&]() {
    copy(a, b);
  });
  assert_used(b);

  array<int, interleaved_shape> c({width, height, depth});
  ASSERT_EQ(c.shape().flat_extent(), c.size());
  double memcpy_time = time_ms([&] {
    memcpy(&c(0, 0, 0), &a(0, 0, 0), a.size() * sizeof(int));
  });
  assert_used(c);

  // copy should be about as fast as memcpy.
  ASSERT_REQ(copy_time, memcpy_time, memcpy_time * 0.2f);
}

TEST(performance_for_each_value) {
  int width = not_constant(500);
  int height = not_constant(200);
  int depth = not_constant(100);

  shape_of_rank<3> s(width, height, depth);
  s.dim<2>().set_stride(1);
  s.dim<1>().set_stride(depth);
  s.dim<0>().set_stride(height * depth);
  ASSERT_EQ(s.flat_extent(), s.size());

  int counter = 0;

  array_of_rank<int, 3> a(s);
  double loop_time = time_ms([&]() {
    for (int z = 0; z < depth; z++) {
      for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	  a(x, y, z) = counter++;
	}
      }
    }
  });
  assert_used(a);

  counter = 0;

  array_of_rank<int, 3> b(s);
  double for_each_value_time = time_ms([&]() {
    b.for_each_value([&](int& x) { x = counter++; });
  });
  assert_used(b);

  // The optimized for_each_value should be quite a bit faster.
  ASSERT_LT(for_each_value_time, loop_time / 2);
}

}  // namespace array

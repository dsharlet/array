#include "array.h"
#include "test.h"

#include <cmath>
#include <chrono>

namespace array {

// Benchmark a call.
template <typename F>
double time_ms(F op) {
  op();
  auto t1 = std::chrono::high_resolution_clock::now();
  op();
  auto t2 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
}

// Tricks the compiler into not stripping away dead objects.
template <typename T>
__attribute__((noinline)) void AssertUsed(const T& x) {}

TEST(set_dense3d_performance) {
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
  AssertUsed(a);

  // Set every value using for_all_indices.
  dense_array<int, 3> b({width, height, depth});
  double for_all_indices_time = time_ms([=, &b]() {
    for_all_indices(b.shape(), [&](int x, int y, int z) {
      b(x, y, z) = z * width * height + y * width + x;
    });
  });
  AssertUsed(b);

  // Set every value using for_each_index.
  dense_array<int, 3> c({width, height, depth});
  double for_each_index_time = time_ms([=, &c]() {
    for_each_index(c.shape(), [&](const dense_array<int, 3>::index_type& i) {
      c(i) = std::get<2>(i) * width * height + std::get<1>(i) * width + std::get<0>(i);
    });
  });
  AssertUsed(c);

  // Check that timings are acceptable. These are actually a lot faster.
  const double tolerance = loop_time * 0.25f;
  ASSERT_REQ(for_all_indices_time, loop_time * 0.66f, tolerance);
  ASSERT_REQ(for_each_index_time, loop_time * 0.66f, tolerance);
}

TEST(copy_dense3d_performance) {
  int width = 500;
  int height = 200;
  int depth = 100;

  dense_array<int, 3> a({width, height, depth}, 3);

  dense_array<int, 3> b({width, height, depth});
  double copy_time = time_ms([&]() {
    copy(a, b);
  });
  AssertUsed(b);

  dense_array<int, 3> c({width, height, depth});
  ASSERT_EQ(c.shape().flat_extent(), c.size());
  double memcpy_time = time_ms([&] {
    memcpy(&c(0, 0, 0), &a(0, 0, 0), a.size() * sizeof(int));
  });
  AssertUsed(c);

  ASSERT_REQ(copy_time, memcpy_time, memcpy_time * 0.5f);
}

}  // namespace array

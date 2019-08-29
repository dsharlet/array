#include "array.h"
#include "test.h"
#include "lifetime.h"

#include <vector>

namespace array {

TEST(array_ref_indices) {
  std::vector<int> data(100);
  for (std::size_t i = 0; i < data.size(); i++) {
    data[i] = i;
  }

  dense_array_ref<int, 1> ref_1d(data.data(), make_dense_shape(100));
  for_all_indices(ref_1d.shape(), [&](int x) {
    ASSERT_EQ(ref_1d(x), x);
  });

  dense_array_ref<int, 2> ref_2d(data.data(), make_dense_shape(20, 5));
  for_all_indices(ref_2d.shape(), [&](int x, int y) {
    ASSERT_EQ(ref_2d(x, y), y*20 + x);
  });
}

TEST(reinterpret) {
  float eight = 8.0f;
  int eight_int;
  ASSERT_EQ(sizeof(eight), sizeof(eight_int));
  memcpy(&eight, &eight_int, sizeof(eight));

  dense_array<int, 3> int_array({4, 5, 6}, eight_int);
  auto float_array = int_array.reinterpret<float>();
  for_all_indices(int_array.shape(), [&](int x, int y, int z) {
    ASSERT_EQ(int_array(x, y, z), eight_int);
    ASSERT_EQ(float_array(x, y, z), eight);
  });
}

}  // namespace array

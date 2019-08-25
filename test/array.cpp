#include "array.h"
#include "test.h"

namespace array {

TEST(array_1d) {
  auto A = make_dense_array<int>(10);
  for_all_indices(A.shape(), [&](int x) {
    A(x) = x;
  });

  auto B = make_dense_array<int>(10);
  B = A;
  for_all_indices(B.shape(), [&](int x) {
    ASSERT_EQ(B(x), x);
  });
}

TEST(array_2d) {
  auto A = make_dense_array<int>(10, 5);
  for_all_indices(A.shape(), [&](int x, int y) {
    A(x, y) = y * 100 + x;
  });

  auto B = make_dense_array<int>(10, 5);
  B = A;
  for_all_indices(B.shape(), [&](int x, int y) {
    ASSERT_EQ(B(x, y), y * 100 + x);
  });  
}

}  // namespace array

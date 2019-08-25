#include "array.h"
#include "test.h"

namespace array {

typedef shape<dim<>, dense_dim<>> matrix_shape;

template <typename T>
using matrix = array<T, shape<dim<>, dense_dim<>>>;

template <typename T>
matrix<T> make_matrix(int M, int N) {
  matrix_shape shape({0, M, N}, {0, N});
  return matrix<T>(shape, 0);
}

template <typename T>
index_t rows(const matrix<T>& m) { return m.shape().template dim<0>().extent(); }
template <typename T>
index_t cols(const matrix<T>& m) { return m.shape().template dim<1>().extent(); }

template <typename T>
void multiply_naive(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  for (int i = 0; i < rows(c); i++) {
    for (int j = 0; j < cols(c); j++) {
      c(i, j) = 0;
      for (int k = 0; k < cols(a); k++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

template <typename T>
void multiply(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  for (int i = 0; i < rows(c); i++) {
    for (int j = 0; j < cols(c); j++) {
      c(i, j) = 0;
    }
    for (int k = 0; k < cols(a); k++) {
      for (int j = 0; j < cols(c); j++) {
        c(i, j) += a(i, k) * b(k, j);
      }
    }
  }
}

TEST(matrix_multiply) {
  auto a = make_matrix<float>(10, 20);
  auto b = make_matrix<float>(20, 30);
  auto c = make_matrix<float>(10, 30);
  a.for_each_value([](float& x) { x = rand(); });
  b.for_each_value([](float& x) { x = rand(); });
  multiply(a, b, c);

  auto c_ref = make_matrix<float>(10, 30);
  multiply_naive(a, b, c_ref);
  
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 30; j++) {
      ASSERT_EQ(c(i, j), c_ref(i, j));
    }
  }
}

}  // namespace array

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
const dim<>& rows(const matrix<T>& m) { return m.shape().template dim<0>(); }
template <typename T>
const dense_dim<>& cols(const matrix<T>& m) { return m.shape().template dim<1>(); }

template <typename T>
void multiply_naive(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  for (int i : rows(c)) {
    for (int j : cols(c)) {
      T c_ij = 0;
      for (int k : cols(a)) {
        c_ij += a(i, k) * b(k, j);
      }
      c(i, j) = c_ij;
    }
  }
}

template <typename T>
void multiply(const matrix<T>& a, const matrix<T>& b, matrix<T>& c) {
  for (int i : rows(c)) {
    for (int j : cols(c)) {
      c(i, j) = 0;
    }
    for (int k : cols(a)) {
      for (int j : cols(c)) {
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

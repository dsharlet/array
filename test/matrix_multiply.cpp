#include "array.h"
#include "test.h"

namespace array {

template <typename T>
using matrix = array<T, shape<dim<>, dense_dim<>>>;

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
  matrix<float> a({10, 20});
  matrix<float> b({20, 30});
  matrix<float> c({10, 30});
  a.for_each_value([](float& x) { x = rand(); });
  b.for_each_value([](float& x) { x = rand(); });
  multiply(a, b, c);

  matrix<float> c_ref({10, 30});
  multiply_naive(a, b, c_ref);
  
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 30; j++) {
      ASSERT_EQ(c(i, j), c_ref(i, j));
    }
  }
}

}  // namespace array

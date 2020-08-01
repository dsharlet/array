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

#ifndef NDARRAY_RATIONAL_H
#define NDARRAY_RATIONAL_H

// Signed integer division in C/C++ is terrible. These implementations
// of Euclidean division and mod are taken from:
// https://github.com/halide/Halide/blob/1a0552bb6101273a0e007782c07e8dafe9bc5366/src/CodeGen_Internal.cpp#L358-L408
template <typename T>
T euclidean_div(T a, T b) {
  T q = a / b;
  T r = a - q * b;
  T bs = b >> (sizeof(T) * 8 - 1);
  T rs = r >> (sizeof(T) * 8 - 1);
  return q - (rs & bs) + (rs & ~bs);
}

template <typename T>
T euclidean_mod(T a, T b) {
  T r = a % b;
  T sign_mask = r >> (sizeof(T) * 8 - 1);
  return r + (sign_mask & std::abs(b));
}

template <typename T>
T gcd(T a, T b) {
  if (b == 0) {
    return a;
  } else {
    return gcd(b, euclidean_mod(a, b));
  }
}

// Represents a rational number. Good for exact arithmetic on fractions.
template <typename T>
class rational {
  T n, d;

  void reduce() {
    T r = gcd(n, d);
    n /= r;
    d /= r;
  }

public:
  rational(T x = 0) : n(x), d(1) {}
  rational(T n, T d) : n(n), d(d) { reduce(); }

  T numerator() const { return n; }
  T denominator() const { return d; }

  rational operator*(const rational& r) const { return {n * r.n, d * r.d}; }
  rational operator/(const rational& r) const { return {n * r.d, d * r.n}; }
  rational operator+(const rational& r) const { return {n * r.d + r.n * d, d * r.d}; }
  rational operator-(const rational& r) const { return {n * r.d - r.n * d, d * r.d}; }

  bool operator<(const rational& r) const { return n * r.d < r.n * d; }
  bool operator<=(const rational& r) const { return n * r.d <= r.n * d; }
  bool operator>(const rational& r) const { return n * r.d > r.n * d; }
  bool operator>=(const rational& r) const { return n * r.d >= r.n * d; }
  bool operator==(const rational& r) const { return n * r.d == r.n * d; }
  bool operator!=(const rational& r) const { return n * r.d != r.n * d; }
};

template <typename T>
rational<T> operator*(T l, const rational<T>& r) {
  return rational<T>(l) * r;
}
template <typename T>
rational<T> operator/(T l, const rational<T>& r) {
  return rational<T>(l) / r;
}
template <typename T>
rational<T> operator-(T l, const rational<T>& r) {
  return rational<T>(l) - r;
}
template <typename T>
rational<T> operator+(T l, const rational<T>& r) {
  return rational<T>(l) + r;
}

template <typename T>
rational<T> min(const rational<T>& l, const rational<T>& r) {
  return l < r ? l : r;
}
template <typename T>
rational<T> max(const rational<T>& l, const rational<T>& r) {
  return l > r ? l : r;
}

template <typename T>
T floor(const rational<T>& r) {
  return euclidean_div(r.numerator(), r.denominator());
}
template <typename T>
T round(const rational<T>& r) {
  return euclidean_div(2 * r.numerator() + r.denominator(), 2 * r.denominator());
}
template <typename T>
T ceil(const rational<T>& r) {
  return euclidean_div(r.numerator() + r.denominator() - 1, r.denominator());
}
template <typename T>
rational<T> frac(const rational<T>& r) {
  return {euclidean_mod(r.numerator(), r.denominator()), r.denominator()};
}
template <typename T>
float to_float(const rational<T>& r) {
  return static_cast<float>(r.numerator()) / r.denominator();
}

#endif // NDARRAY_RATIONAL_H

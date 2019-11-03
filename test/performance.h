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
__attribute__((noinline)) void assert_used(const T& x) {}

// Tricks the compiler into not constant folding the result of x.
template <typename T>
__attribute__((noinline)) T not_constant(T x) { return x; }

}  // namespace array

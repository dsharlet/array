#ifndef ARRAY_BENCHMARK_H
#define ARRAY_BENCHMARK_H

#include <cmath>
#include <chrono>

namespace array {

// Benchmark a call.
template <typename F>
double benchmark(int samples, int iterations, F op) {
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < samples; i++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < iterations; j++) {
            op();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / (iterations*1e9);
        if (dt < best)
            best = dt;
    }
    return best;
}

}  // namespace array

#endif

// Workarounds so that we don't need to install the CUDA toolkit.
// This is just a build test.
#if defined(__CUDA__)
#ifndef __device__
#define __device__ __attribute__((device))
#endif

#ifndef __host__
#define __host__ __attribute__((host))
#endif

// For some reason, this doesn't compile when it's size_t.
__device__ void* malloc(int size);
__device__ void free(void* p);

#endif  // defined(__CUDA__)

#include "array.h"

namespace nda {

NDARRAY_HOST_DEVICE
void array_ref_indices() {
  int data[100];
  for (int i = 0; i < 100; i++) {
    data[i] = i;
  }
  dense_array_ref<int, 1> ref_1d(data, {100});
}

NDARRAY_HOST_DEVICE
void reinterpret() {
  float eight = 8.0f;
  int eight_int = *reinterpret_cast<int*>(&eight);

  dense_array_ref<int, 3> int_array(&eight_int, {1, 1, 1});
  dense_array_ref<float, 3> float_array = reinterpret<float>(int_array);
  (void)int_array;
  (void)float_array;
}

NDARRAY_HOST_DEVICE
void array_ref_empty() {
  // This does *not* work: it that shape_ = new_shape is not allowed because we use the
  // defaulted assignment operator, which is apparently __host__ only? This seems
  // like a bug in clang, the operator is explicitly defaulted with a __device__ annotation:
  //   NDARRAY_HOST_DEVICE
  //   shape& operator=(const shape&) = default;
  //
  // dense_array_ref<int, 1> null_ref(nullptr, {10});
  // null_ref.set_shape({{3, 3}}, 3);

  int x;
  array_ref_of_rank<int, 0> scalar_ref(&x, {});
  array_ref_of_rank<int, 0> null_scalar_ref(nullptr, {});
}

// TODO(jiawen): Add CUDA support to image.h, matrix.h, and einsum.h.

}  // namespace nda

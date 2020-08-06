## About

This library provides a multidimensional array class for C++, with the following design goals:
* Enable specification of array parameters as [compile-time constants](#compile-time-constant-shapes) per parameter, enabling more efficient code generation, while retaining run-time flexibility where needed.
* Provide an API following the conventions of the C++ STL where possible.
* Minimal dependencies and requirements (the library is currently a single header file, and depends only on the C++ STL).

The library uses some ideas established in other existing projects, such as [numpy](https://numpy.org/doc/1.17/reference/arrays.ndarray.html), [Halide](https://halide-lang.org/docs/class_halide_1_1_runtime_1_1_buffer.html), and [Eigen](http://eigen.tuxfamily.org).
Array shapes are specified as a list of N dimensions, where each dimension has parameters such as an extent and a stride.
Array references and objects use shape objects to map N-dimensional indices to a flat index.
N-dimensional indices are mapped to flat offsets with the following formula:
```
flat_offset = (x0 - min0)*stride0 + (x1 - min1)*stride1 + ... + (xN - minN)*strideN
```
where:
* `xN` are the indices in each dimension.
* `minN` are the mins in each dimension. The min is the value of the first in-range index in this dimension (the max is `minN + extentN - 1`).
* `strideN` are the distances in the flat offsets between elements in each dimension.

Arrays efficiently support advanced manipulations like [cropping, slicing, and splitting](#slicing-cropping-and-splitting) arrays and loops, all while preserving compile-time constant parameters when possible.
Although it is a heavily templated library, most features do not have significant code size or compile time implications, and incorrect usage generates informative and helpful error messages.
Typically, an issue will result in only one error message, located at the site of the problem in user code.

Many other libraries offering multi-dimensional arrays or tensors allow compile-time constant shapes.
*However*, most if not all of them only allow either all of the shape parameters to be compile-time constant, or none of them.
This is really limiting; often only a few key parameters of a shape need to be compile-time constant for performance, while other dimensions need flexibility to accommodate runtime-valued shape parameters.
Some examples of this are:
* '[Chunky](https://en.wikipedia.org/wiki/Packed_pixel)' image formats with a small fixed number of channels.
* Matrices where one dimension represent variables intrinsic to the problem, while the other dimension represents a number of samples of data.
* Algorithms optimized by splitting or tiling intermediate stages will have intermediate buffers that have a constant extent in the dimensions that are split or tiled.

Some other features of the library are:
* [CUDA support](#cuda-support) for use in `__device__` functions.
* [Einstein reduction](#einstein-reductions) helpers, enabling many kinds of reductions and other array operations to be expressed safely.

## Usage

### Shapes

The basic types provided by the library are:
* `dim<Min, Extent, Stride>`, a description of a single dimension. The template parameters specify a compile-time constant min, extent, or stride, or are `dynamic` (meaning unknown) and are specified at runtime.
* `shape<Dim0, Dim1, ...>`, a description of multiple dimensions. `Dim0` is referred to as the innermost dimension.
* `array<T, Shape, Allocator>`, a container following the conventions of `std::vector` where possible. This container manages the allocation of a buffer associated with a `Shape`.
* `array_ref<T, Shape>`, a wrapper for addressing existing memory with a shape `Shape`.

To define an array, define a shape type, and use it to define an array object:
```c++
  using my_3d_shape_type = shape<dim<>, dim<>, dim<>>;
  constexpr int width = 16;
  constexpr int height = 10;
  constexpr int depth = 3;
  my_3d_shape_type my_3d_shape(width, height, depth);
  array<int, my_3d_shape_type> my_array(my_3d_shape);
```

General shapes and arrays like this have the following built-in aliases:
* `shape_of_rank<N>`, an N-dimensional shape.
* `array_ref_of_rank<T, N>` and `array_of_rank<T, N, Allocator>`, N-dimensional arrays with a shape of `shape_of_rank<N>`.

### Access and iteration

Accessing `array` or `array_ref` is done via `operator(...)` and `operator[index_type]`.
There are both variadic and `index_type` overloads of `operator()`.
`index_type` is a specialization of `std::tuple` defined by `shape` (and `array` and `array_ref`), e.g. `my_3d_shape_type::index_type`.
```c++
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // Variadic verion:
        my_array(x, y, z) = 5;
        // Or the index_type versions:
        my_array({x, y, z}) = 5;
        my_array[{x, y, z}] = 5;
      }
    }
  }
```

`array::for_each_value` and `array_ref::for_each_value` calls a function with a reference to each value in the array.
```c++
  my_array.for_each_value([](int& value) {
    value = 5;
  });
```

`for_all_indices` is a free function taking a shape object and a function to call with every index in the shape.
`for_each_index` is similar, calling a free function with the index as an instance of the index type `my_3d_shape_type::index_type`.
```c++
  for_all_indices(my_3d_shape, [&](int x, int y, int z) {
    my_array(x, y, z) = 5;
  });
  for_each_index(my_3d_shape, [&](my_3d_shape_type::index_type i) {
    my_array[i] = 5;
  });
```

The order in which each of `for_each_value`, `for_each_index`, and `for_all_indices` execute their traversal over the shape is defined by `shape_traits<Shape>`.
The default implementation of `shape_traits<Shape>::for_each_index` iterates over the innermost dimension as the innermost loop, and proceeds in order to the outermost dimension.
```c++
  my_3d_shape_type my_shape(2, 2, 2);
  for_all_indices(my_shape, [](int x, int y, int z) {
    std::cout << x << ", " << y << ", " << z << std::endl;
  });
  // Output:
  // 0, 0, 0
  // 1, 0, 0
  // 0, 1, 0
  // 1, 1, 0
  // 0, 0, 1
  // 1, 0, 1
  // 0, 1, 1
  // 1, 1, 1
```

The default implementation of `shape_traits<Shape>::for_each_value` iterates over a dynamically optimized shape.
The order will vary depending on the properties of the shape.

There are overloads of `for_all_indices` and `for_each_index` accepting a permutation to indicate the loop order. In this example, the permutation `<2, 0, 1>` iterates over the `z` dimension as the innermost loop, then `x`, then `y`.
```c++
  for_all_indices<2, 0, 1>(my_shape, [](int x, int y, int z) {
    std::cout << x << ", " << y << ", " << z << std::endl;
  });
  // Output:
  // 0, 0, 0
  // 0, 0, 1
  // 1, 0, 0
  // 1, 0, 1
  // 0, 1, 0
  // 0, 1, 1
  // 1, 1, 0
  // 1, 1, 1
```

### Compile-time constant shapes

In the previous examples, no array parameters are compile time constants, so all of these accesses and loops expand to a `flat_offset` expression where the mins, extents, and strides are runtime variables.
This can prevent the compiler from generating efficient code.
For example, the compiler may be able to auto-vectorize these loops, but if the stride of the dimension accessed by the vectorized loop is a runtime variable, the compiler will have to generate gathers and scatters instead of vector load and store instructions, even if the stride is one at runtime.

To avoid this, we need to make array parameters compile time constants.
However, while making array parameters compile time constants helps the compiler generate efficient code, it also makes the program less flexible.

This library helps balance this tradeoff by enabling any of the array parameters to be compile time constants, but not require it.
Which parameters should be made into compile time constants will vary depending on the use case.
A common case is to make the innermost dimension have stride 1:
```c++
  using my_dense_3d_shape_type = shape<
      dim</*Min=*/dynamic, /*Extent=*/dynamic, /*Stride=*/1>,
      dim<>,
      dim<>>;
  array<char, my_dense_3d_shape_type> my_dense_array({16, 3, 3});
  for (auto x : my_dense_array.x()) {
    // The compiler knows that each loop iteration accesses
    // elements that are contiguous in memory for contiguous x.
    my_dense_array(x, y, z) = 0;
  }
```

A dimension with unknown min and extent, and stride 1, is common enough that it has a built-in alias `dense_dim<>`, and shapes with a dense first dimension are common enough that shapes and arrays have the following built-in aliases:
* `dense_shape<N>`, an N-dimensional shape with the first dimension being dense.
* `dense_array_ref<T, N>` and `dense_array<T, N, Allocator>`, N-dimensional arrays with a shape of `dense_shape<N>`.

There are other common examples that are easy to support.
A very common array is an image where 3-channel RGB or 4-channel RGBA pixels are stored together in a 'chunky' format.
```c++
template <int Channels, int PixelStride = Channels>
using chunky_image_shape = shape<
    strided_dim</*Stride=*/PixelStride>,
    dim<>,
    dense_dim</*Min=*/0, /*Extent=*/Channels>>;
```

`strided_dim<>` is another alias for `dim<>` where the min and extent are unknown, and the stride may be a compile-time constant.
[`image.h`](image.h) is a small helper library of typical image shape and object types defined using arrays, including `chunky_image_shape`.

Another common example is matrices indexed `(row, column)` with the column dimension stored densely:
```c++
  using matrix_shape = shape<dim<>, dense_dim<>>;
  array<double, matrix_shape> my_matrix({10, 4});
  for (auto i : my_matrix.i()) {
    for (auto j : my_matrix.j()) {
      // This loop ordering is efficient for this type.
      my_matrix(i, j) = 0.0;
    }
  }
```

There are also many use cases for matrices with small constant sizes.
This library provides `auto_allocator<T, N>`, an `std::allocator` compatible allocator that only allocates buffers of `N` small fixed sized objects with automatic storage.
This makes it possible to define a small matrix type that will not use any dynamic memory allocation:
```c++
template <int M, int N>
using small_matrix_shape = shape<
    dim<0, M>,
    dense_dim<0, N>>;
template <typename T, int M, int N>
using small_matrix = array<T, small_matrix_shape<M, N>, auto_allocator<T, M*N>>;
small_matrix<float, 4, 4> my_small_matrix;
// my_small_matrix is only one fixed size allocation, no new/delete calls
// happen. sizeof(small_matrix) = sizeof(float) * 4 * 4 + (overhead)
```

[`matrix.h`](matrix.h) is a small helper library of typical matrix shape and object types defined using arrays, including the examples above.

### Slicing, cropping, and splitting

Shapes and arrays can be sliced and cropped using `interval<Min, Extent>` objects, which are similar to `dim<>`s.
They can have either a compile-time constant or runtime valued min and extent.
`range(begin, end)` is a helper functions to construct an `interval`.
```c++
  // Slicing
  array_ref_of_rank<int, 2> channel1 = my_array(_, _, 1);
  array_ref_of_rank<int, 1> row4_channel2 = my_array(_, 4, 2);

  // Cropping
  array_ref_of_rank<int, 3> top_left = my_array(interval<>{0, 2}, interval<>{0, 4}, _);
  array_ref_of_rank<int, 2> center_channel0 = my_array(interval<>{1, 2}, interval<>{2, 4}, 0);
```
The `_` or `all` constants are placeholders indicating the entire dimension should be preserved.
Dimensions that are sliced are removed from the shape of the array.

When iterating a `dim`, it is possible to `split` it first by either a compile-time constant or a runtime-valued split factor.
A split `dim` produces an iterator range that produces `interval<>` objects.
This allows easy tiling of algorithms:
```c++
  constexpr index_t x_split_factor = 3;
  const index_t y_split_factor = 5;
  for (auto yo : split(my_array.y(), y_split_factor)) {
    for (auto xo : split<x_split_factor>(my_array.x())) {
      auto tile = my_array(xo, yo, _);
      for (auto x : tile.x()) {
        // The compiler knows this loop has a fixed extent x_split_factor!
        tile(x, y, z) = x;
      }
    }
  }
```

Both loops have extents that are not divided by their split factors.
To avoid generating an `array_ref` referencing data out of bounds of the original array, the split iterators modify the last iteration.
The behavior of each kind of split is different:
* Because the extent of `yo` can vary, it is reduced on the last iteration. This strategy can accommodate dimensions of any extent.
* Because the extent of `xo` must be a constant, the last iteration will be shifted to overlap the previous iteration. This strategy requires the extent of the dimension being split is greater than the split factor (but not a multiple!)

Compile-time constant split factors produce ranges with compile-time extents, and shapes and arrays cropped with these ranges will have a corresponding `dim<>` with a compile-time constant extent.
This allows potentially significant optimizations to be expressed relatively easily!

### Einstein reductions

The [`ein_reduce.h`](ein_reduce.h) header provides [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) reductions and summation helpers, similar to [np.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) or [tf.einsum](https://www.tensorflow.org/api_docs/python/tf/einsum).
These are zero-cost abstractions for describing loops that allow expressing a wide variety of array operations.
Einstein notation expression operands are constructed using the `ein<i, j, ...>(x)` helper function, where `x` can be any callable object, including an `array<>` or `array_ref<>`.
`i, j, ...` are `constexpr` integers indicating which dimensions of the reduction operation are used to evaluate `x`.
Therefore, the number of arguments of `x` must match the number of dimensions provided to `ein`.
Operands can be combined into larger expressions using typical binary operators.

Einstein notation expressions can be evaluated using one of the following functions:
* `ein_reduce(expression)`, evaluate an arbitrary Einstein notation `expression`.
* `lhs = make_ein_sum<T, i, j, ...>(rhs)`, evaluate the summation `ein<i, j, ...>(lhs) += rhs`, and return `lhs`. The shape of `lhs` is inferred from the expression.

Here are some examples using these reduction operations to compute summations:
```c++
  // Name the dimensions we use in Einstein reductions.
  enum { i = 0, j = 1, k = 2, l = 3 };

  // Dot product dot1 = dot2 = x.y:
  vector<float> x({10});
  vector<float> y({10});
  float dot1 = make_ein_sum<float>(ein<i>(x) * ein<i>(y));
  float dot2 = 0.0f;
  ein_reduce(ein(dot2) += ein<i>(x) * ein<i>(y));

  // Matrix multiply C1 = C2 = A*B:
  matrix<float> A({10, 10});
  matrix<float> B({10, 15});
  matrix<float> C1({10, 15});
  fill(C1, 0.0f);
  ein_reduce(ein<i, j>(C1) += ein<i, k>(A) * ein<k, j>(B));
  auto C2 = make_ein_sum<float, i, j>(ein<i, k>(A) * ein<k, j>(B));
```

We can use arbitrary functions as expression operands:
```c++
  // Cross product array crosses_n = x_n x y_n:
  using vector_array = array<float, shape<dim<0, 3>, dense_dim<>>>;
  vector_array xs({3, 100});
  vector_array ys({3, 100});
  vector_array crosses({3, 100});
  auto epsilon3 = [](int i, int j, int k) { return sgn(j - i) * sgn(k - i) * sgn(k - j); };
  ein_reduce(ein<i, l>(crosses) += ein<i, j, k>(epsilon3) * ein<j, l>(xs) * ein<k, l>(ys));
```

These operations generally produce loop nests that are as readily optimized by the compiler as hand-written loops.
For example, consider the cross product: `crosses`, `xs`, and `ys` have shape `shape<dim<0, 3>, dense_dim<>>`, so the compiler will see small constant-range loops and likely be able to optimize this to similar efficiency as hand-written code, by unrolling and evaluating the function at compile time.
The compiler will also likely be able to efficiently vectorize the `l` dimension of the `ein_reduce`, because that dimension has a constant stride 1.

The expression can be another kind of reduction, or not a reduction at all:
```c++
  // Matrix transpose AT = A^T:
  matrix<float> AT({10, 10});
  ein_reduce(ein<i, j>(AT) = ein<j, i>(A));

  // Maximum of each x-y plane of a 3D volume:
  dense_array<float, 3> T({8, 12, 20});
  dense_array<float, 1> max_xy({20});
  auto r = ein<k>(max_xy);
  ein_reduce(r = max(r, ein<i, j, k>(T)));
```

Reductions can have a mix of result and operand types:
```c++
  // Compute X1 = X2 = DFT[x]:
  using complex = std::complex<float>;
  dense_array<complex, 2> W({10, 10});
  for_all_indices(W.shape(), [&](int j, int k) {
    W(j, k) = exp(-2.0f * pi * complex(0, 1) * (static_cast<float>(j * k) / 10));
  });
  auto X1 = make_ein_sum<complex, j>(ein<j, k>(W) * ein<k>(x));
  vector<complex> X2({10}, 0.0f);
  ein_reduce(ein<j>(X2) += ein<j, k>(W) * ein<k>(x));
```

These reductions also compose well with loop transformations like `split` and array operations like [slicing and cropping](#slicing-cropping-and-splitting).
For example, a matrix multiplication can be tiled like so:
```c++
  // Adjust this depending on the target architecture. For AVX2, vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(float);

  // We want the tiles to be big without spilling the accumulators to the stack.
  constexpr index_t tile_rows = 3;
  constexpr index_t tile_cols = vector_size * 3;

  for (auto io : split<tile_rows>(C.i())) {
    for (auto jo : split<tile_cols>(C.j())) {
      auto C_ijo = C(io, jo);
      fill(C_ijo, 0.0f);
      ein_reduce(ein<i, j>(C_ijo) += ein<i, k>(A(io, _)) * ein<k, j>(B(_, jo)));
    }
  }
```

This generates the following machine code(\*) for the inner loop using clang 11 with -O2 -ffast-math:
```assembly
LBB9_11:
        vmovaps %ymm9, %ymm10
        vmovaps %ymm8, %ymm11
        vbroadcastss    (%r12,%rbx,4), %ymm12
        vbroadcastss    (%r13,%rbx,4), %ymm13
        vbroadcastss    (%r10,%rbx,4), %ymm14
        vmovups -64(%r14), %ymm9
        vmovups -32(%r14), %ymm8
        vmovups (%r14), %ymm15
        vfmadd231ps     %ymm12, %ymm9, %ymm5
        vfmadd231ps     %ymm13, %ymm9, %ymm7
        vfmadd213ps     %ymm10, %ymm14, %ymm9
        vfmadd231ps     %ymm12, %ymm8, %ymm2
        vfmadd231ps     %ymm13, %ymm8, %ymm6
        vfmadd213ps     %ymm11, %ymm14, %ymm8
        vfmadd231ps     %ymm12, %ymm15, %ymm1
        vfmadd231ps     %ymm13, %ymm15, %ymm4
        vfmadd231ps     %ymm14, %ymm15, %ymm3
        incq    %rbx
        addq    %rcx, %r14
        cmpq    %rbx, %rsi
        jne     LBB9_11
```
This is **30-40x** faster than a naive C implementation of nested loops on my machine, and I believe it is within a factor of 2 of the peak performance possible.

(\*) Unfortunately, this doesn't generate performant code currently and requires a few tweaks to work around an issue in LLVM.
See the [matrix example](examples/linear_algebra/matrix.cpp) for the example code that produces the above assembly.
To summarise, it is currently necessary to perform the accumulation into a temporary buffer instead of accumulating directly into the output.
I think this will be unnecessary when LLVM fixes a [basic issue](https://bugs.llvm.org/show_bug.cgi?id=45863).

### CUDA support

Most of the functions in this library are marked with `__device__`, enabling them to be used in CUDA code.
This includes `array_ref<T, Shape>` and most of its helper functions.
The exceptions to this are functions and classes that allocate memory, primarily `array<T, Shape, Alloc>`.

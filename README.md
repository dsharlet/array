## About

This library provides a multidimensional array class for C++, with the following design goals:
* Enable specification of array parameters as compile-time constants, enabling significantly more efficient code generation in some cases.
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

## Usage

### Shapes

The basic types provided by the library are:
* `dim<Min, Extent, Stride>`, a description of a single dimension. The template parameters specify a compile-time constant min, extent, or stride, or are `UNK` (the default, meaning unknown) and are specified at runtime.
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

### Compile-time constant shapes

In these examples, no array parameters are compile time constants, so all of these accesses and loops expand to a `flat_offset` expression where the strides are runtime variables.
This can prevent the compiler from generating efficient code.
For example, the compiler may be able to auto-vectorize these loops, but if the stride of the vectorized dimension is a runtime variable, the compiler will have to generate gather and scatter instructions instead of load and store instructions, even if the stride is one.

To avoid this, we need to make array parameters compile time constants.
However, while making array parameters compile time constants helps the compiler generate efficient code, it also makes the program less flexible.

This library helps balance this tradeoff by enabling any of the array parameters to be compile time constants, but not require it.
Which parameters should be made into compile time constants will vary depending on the use case.
A common case is to make the innermost dimension have stride 1:
```c++
using my_dense_3d_shape_type = shape<
    dim</*Min=*/UNK, /*Extent=*/UNK, /*Stride=*/1>,
    dim<>,
    dim<>>;
```

A dimension with unknown min and extent, and stride 1, is common enough that it has a built-in alias `dense_dim<>`, and shapes with a dense first dimension are common enough that they have the following built-in aliases:
* `dense_shape<N>`, an N-dimensional shape with the first dimension being dense.
* `dense_array_ref<T, N>` and `dense_array<T, N, Allocator>`, N-dimensional arrays with a shape of `dense_shape<N>`.

There are other common examples that are easy to support.
A very common array is an image where 3-channel RGB or 4-channel RGBA pixels are stored together in a 'chunky' format.
```c++
template <int Channels, int ChannelStride = Channels>
using chunky_image_shape = shape<
    strided_dim</*Stride=*/ChannelStride>,
    dim<>,
    dense_dim</*Min=*/0, /*Extent=*/Channels>>;
```

`strided_dim<>` is another alias for `dim<>` where the min and extent are unknown, and the stride may be a compile-time constant.
`image.h` is a small helper library of typical image shape and object types defined using arrays, including `chunky_image_shape`.

Another common example is matrices indexed `(row, column)` with the column dimension stored densely:
```c++
using matrix_shape = shape<dim<>, dense_dim<>>;
```

There are also many use cases for matrices with small constant sizes.
This library provides `stack_allocator<T, N>`, an `std::allocator` compatible allocator that only allocates buffers of `N` small fixed sized objects.
This makes it possible to define a small matrix type that will not use any dynamic memory allocation:
```c++
template <int M, int N>
using small_matrix_shape = shape<
    dim<0, M>,
    dense_dim<0, N>>;
template <typename T, int M, int N>
using small_matrix = array<T, small_matrix_shape, stack_allocator<T, M*N>>;
```

### Slicing, cropping, and splitting

Shapes and arrays can be sliced and cropped using `range<Min, Extent>` objects, which are similar to `dim<>`s.
They can have either a compile-time constant or runtime valued min and extent.
```c++
my_3d_shape_type my_shape(4, 8, 3);
array<int, my_3d_shape_type> my_array(my_shape);
// Slicing
my_3d_shape_type sliced_shape = shape(_, _, 0);
array_ref<int, my_3d_shape_type> sliced_array = my_array(_, _, 1);

// Cropping
my_3d_shape_type top_left_shape = my_shape(range<>{0, 2}, range<>{0, 4}, _);
array_ref<int, my_3d_shape_type> center_crop = my_array(range<>{1, 2}, range<>{2, 4}, _);
```
The `_` constant is a placeholder indicating the entire dimension should be preserved.
When slicing, arrays do not lose rank: the sliced dimension remains with extent 1.

When iterating a `dim`, it is possible to `split` it first by either a compile-time constant or a runtime-valued split factor.
A split `dim` produces an iterator range that produces `range<>` objects.
This allows easy tiling of algorithms:
```c++
array<int, my_3d_shape_type> my_array(16, 12, 3);
constexpr index_t x_split_factor = 3;
const index_t y_split_factor = 5;
for (auto yo : split(my_array.y(), y_split_factor)) {
  for (auto xo : split<x_split_factor>(my_array.x()) {
    auto tile = my_array(xo, yo, _);
    ...
  }
}
```

The behavior of the last iteration in these cases is different.
The last iteration of `xo` will overlap the previous iteration, because the extent of the range must be a constant.
This also requires the extent of the dimension being split to be greater than the split factor.
Because the extent of the `yo` split can vary, it is reduced on the last iteration to accomodate the range with an extent not divided by the split factor.

Note that compile-time constant split factors produce ranges with compile-time extents, and shapes and arrays cropped with these ranges will have a corresponding `dim<>` with a compile-time constant extent.
This allows potentially significant optimizations to be expressed relatively easily.

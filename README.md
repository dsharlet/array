About
-----

This library provides a multidimensional array class for C++, with the following design goals:
* Enabling specification of array parameters as compile-time constants, enabling significantly more efficient code generation in some cases.
* Providing an API following the conventions of the C++ STL where possible.
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

Usage
-----

The basic types provided by the library are:
* `dim<Min, Extent, Stride>`, a description of a single dimension. The template parameters specify compile-time constant mins, extents, and strides, or are `UNK` (the default, meaning unknown) and are specified at runtime.
* `shape<Dim0, Dim1, ...>`, a description of multiple dimensions. `Dim0` is referred to as the innermost dimension.
* `array<T, Shape, Allocator>`, a container following the conventions of `std::vector` where possible. This container manages the allocation of a buffer associated with a `Shape`.
* `array_ref<T, Shape>`, a wrapper for addressing existing memory with a shape 'Shape'.

To define an array, define a shape type, and use it to define an array object:
```c++
typedef shape<dim<>, dim<>, dim<>> my_3d_shape_type;
constexpr int width = 16;
constexpr int height = 10;
constexpr int depth = 3;
my_3d_shape_type my_3d_shape(width, height, depth);
array<int, my_3d_shape> my_array(my_3d_shape);
```

The array can be accessed in a number of ways.
`array::operator()(...)` and `array::at(...)` have similar semantics to `std::vector::operator[]` and `std::vector::at`.
`array::at` will check that the index is in range, and throws `std::out_of_range` if it is not.
There are both variadic and `index_type` overloads of both of these accessors.
`index_type` is a specialization of `std::tuple` defined by `shape<>`, e.g. `my_3d_shape_type::index_type`.
```c++
for (int z = 0; z < depth; z++) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Variadic verions:
      my_array.at(x, y, z) = 5;
      my_array(x, y, z) = 5;
      // Or the index_type versions:
      my_array.at(my_3d_shape_type::index_type(x, y, z)) = 5;
      my_array({x, y, z}) = 5;
    }
  }
}
```

`array::for_each_value` calls a function with a reference to each value in the array.
The order in which `for_each_value` calls the function with references is undefined, allowing the implementation to reorder the traversal to optimize memory access patterns.
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
  my_array(i) = 5;
});
```

The innermost dimension of the shape corresponds to the innermost loop, and subsequent loops are nested outside.
```c++
my_3d_shape_type shape(2, 2, 2);
for_all_indices(shape, [](int x, int y, int z) {
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

`permute<D0, D1, ..., DN>` is a helper function that enables reordering the dimensions of a shape, and can be used to control the order in which loops are executed.
`D0, D1, ..., DN` is a permutation of the dimension indices.
We can change the order of the loops with a `permute` operation (note the reordered indices as well):
```c++
for_all_indices(permute<2, 0, 1>(shape), [](int z, int x, int y) {
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

In these examples, no array parameters are compile time constants, so all of these accesses and loops expand to a `flat_offset` expression where the strides are runtime variables.
This can prevent the compiler from generating efficient code.
For example, the compiler may be able to auto-vectorize these loops, but if the stride of the vectorized dimension is a runtime variable, the compiler will have to generate a gather instead of a load instruction, even if the stride is one.
To avoid this, we need to make array parameters compile time constants.
However, while making array parameters compile time constants helps the compiler generate efficient code, it also makes the program less flexible.
This library helps balance these two conflicting goals by enabling any of the array parameters to be compile time constants, but not require it.
Which parameters should be made into compile time constants will vary depending on the use case.
A common case is to make the innermost dimension have stride 1:
```c++
typedef shape<dim</*Min=*/UNK, /*Extent=*/UNK, /*Stride=*/1>, dim<>, dim<>> my_dense_3d_shape_type;
```

A dimension with unknown min and extent, and stride 1, is common enough that it has a built-in alias `dense_dim<>`, and shapes with a dense first dimension are common enough that they have the following built-in aliases:
* `dense_shape<N>`, an N-dimensional dense shape, with the first dimension being dense.
* `dense_array_ref<T, N>` and `dense_array<T, N, Allocator>`, N-dimensional arrays with a shape of `dense_shape<N>`.

There are other common examples that are easy to support.
A very common array is an image where 3-channel RGB or 4-channel RGBA pixels are stored together in a 'chunky' format.
```c++
template <int Channels>
using chunky_image_shape = shape<dim<UNK, UNK, Channels>, dim<>, dense_dim<0, Channels>>;
```

Another common example is matrices indexed `(row, column)` with the column dimension stored densely:
```c++
using matrix_shape = shape<dim<>, dense_dim<>>;
```

There are also many use cases for matrices with small constant sizes.
This library provides `stack_allocator<T, N>`, an `std::allocator` compatible allocator that only allocates buffers of `N` small fixed sized objects.
This makes it possible to define a small matrix type that will not use any dynamic memory allocation:
```c++
template <int M, int N>
using small_matrix_shape = shape<dim<0, M>, dense_dim<0, N>>;
template <typename T, int M, int N>
using small_matrix = array<T, small_matrix_shape, stack_allocator<T, M*N>>;
```

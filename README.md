About
-----

This library provides a multidimensional array class for C++, with the following design goals:
* Enabling specification of array parameters as compile-time constants, enabling significantly more efficient code generation in some cases.
* Providing an API following the conventions of the C++ STL where possible.
* Minimal dependencies and requirements (the library is currently a single header file).

The library uses some ideas established in other existing projects, such as numpy and Halide.
Array shapes are specified as a list of N dimensions, where each dimension has parameters such as an extent and a stride.
Array references and objects use shape objects to map N-dimensional indices to a flat index.
A number of helper functions exist to support manipulating and traversing shapes and arrays.

Usage
-----

The basic types provided by the library are:
* `dim<Min, Extent, Stride>`, a description of a single dimension. The template parameters specify compile-time constant strides, or are unknown and are specified at runtime.
* `shape<Dim0, Dim1, ...>`, a description of multiple dimensions.
* `array<T, Shape, Allocator>`, a container following the conventions of `std::vector` where possible. This container manages the allocation of a buffer associated with a `Shape`.
* `array_ref<T, Shape>`, a wrapper for addressing existing memory with a shape of the Shape object.

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
There are both variadic and `std::tuple` overloads of both of these accessors.
```c++
for (int z = 0; z < depth; z++) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      my_array.at(x, y, z) = 5;
      my_array(x, y, z) = 5;
      my_array.at(std::make_tuple(x, y, z)) = 5;
      my_array(std::make_tuple(x, y, z)) = 5;
    }
  }
}
```

`array::for_each_value`, calls a function with a reference to each value in the array.
```c++
my_array.for_each_value([](int& value) {
  value = 5;
});
```

`for_all_indices` is a free function taking a shape object and a function to call with every index in the shape.
`for_each_index` is similar, calling a free function with the index as a tuple.
```c++
for_all_indices(my_3d_shape, [&](int x, int y, int z) {
  my_array(x, y, z) = 5;
});
for_each_index(my_3d_shape, [&](std::tuple<int, int, int> i) {
  my_array(i) = 5;
});
```

In this example, no array parameters are compile time constants, so all of these accesses and loops expand to an expression such as `x*Sx + y*Sy + z*Sz`, where `Sn` is a runtime variable describing the stride of dimension `n`. 
This can prevent the compiler from generating efficient code.
To avoid this, we need to make array parameters compile time constants.
However, while making array parameters compile time constants helps the compiler generate efficient code, it also makes the program less flexible.
This library helps balance these two conflicting goals by enabling any of the array parameters to be compile time constants, but not require it.
Which parameters should be made into compile time constants will vary depending on the use case.
A common case, such as the above examples, is to make the innermost dimension have stride 1:
```c++
typedef shape<dim</*MIN=*/UNK, /*EXTENT=*/UNK, /*STRIDE=*/1>, dim<>, dim<>>
```

A dimension with unknown min and extent, and stride 1, is common enough that it has a builtin alias `dense_dim<>`.
A very common kind of array is an image where 3-channel RGB or 4-channel RGBA pixels are stored together.
```c++
template <typename T, int Channels>
using interleaved_image_shape_type =
   shape<dim<UNK, UNK, Channels>, dim<>, dense_dim<0, Channels>>;
```


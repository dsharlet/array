#ifndef ARRAY_H
#define ARRAY_H

#include <memory>

#include "shape.h"

namespace array {

/** A multi-dimensional array container that mirrors std::vector. */
template <typename T, typename Shape, typename Alloc = std::allocator<T>>
class array {
  Alloc alloc_;
  T* base_;
  Shape shape_;

  // After allocate or reallocate, the array is allocated but uninitialized.
  void allocate() {
    if (!base_) {
      base_ = alloc_.allocate(shape_.flat_extent());
    }
  }
  void reallocate(Shape shape) {
    if (shape_ != shape) {
      deallocate();
      shape_ = std::move(shape);
    }
    allocate();
  }

  // deallocate assumes the array has been destroyed.
  void deallocate() {
    if (base_) {
      alloc_.deallocate(base_, shape_.flat_extent());
      base_ = nullptr;
    }
  }

  void construct() {
    assert(base_);
    for_each_value([&](T& x) {
      std::allocator_traits<Alloc>::construct(alloc_, &x);
    });
  }
  void construct(const T& init) {
    assert(base_);
    for_each_value([&](T& x) {
      std::allocator_traits<Alloc>::construct(alloc_, &x, init);
    });
  }
  void construct(const array& copy) {
    assert(base_);
    for_each_index(shape(), [&](const index_type& index) {
      std::allocator_traits<Alloc>::construct(alloc_, &operator()(index), copy(index));
    });
  }
  void construct(array&& move) {
    assert(base_);
    for_each_index(shape(), [&](const index_type& index) {
      std::allocator_traits<Alloc>::construct(alloc_, &operator()(index), std::move(move(index)));
    });
  }
  void destroy() {
    assert(base_);
    for_each_value([&](T& x) {
      std::allocator_traits<Alloc>::destroy(alloc_, &x);
    });
  }

 public:
  typedef T value_type;
  typedef Alloc allocator_type;
  typedef typename Shape::index_type index_type;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef typename std::allocator_traits<Alloc>::pointer pointer;
  typedef typename std::allocator_traits<Alloc>::const_pointer const_pointer;

  array() : base_(nullptr) {}
  explicit array(const Alloc& alloc) : alloc_(alloc), base_(nullptr) {}
  array(Shape shape, const T& value, const Alloc& alloc = Alloc()) : array(alloc) {
    assign(std::move(shape), value);
  }
  explicit array(Shape shape, const Alloc& alloc = Alloc()) : array(alloc) {
    assign(std::move(shape), T());
  }
  array(const array& copy)
      : array(std::allocator_traits<Alloc>::select_on_container_copy_construction(copy.get_allocator())) {
    assign(copy);
  }
  array(const array& copy, const Alloc& alloc) : array(alloc) {
    assign(copy);
  }
  array(array&& other) : array() {
    swap(other);
  }
  array(array&& other, const Alloc& alloc) : array(alloc) {
    using std::swap;
    swap(shape_, other.shape_);
    swap(base_, other.base_);
  }
  ~array() { 
    destroy();
    deallocate(); 
  }

  array& operator=(const array& other) {
    if (this == &other) return *this;

    if (std::allocator_traits<Alloc>::propagate_on_container_copy_assignment::value) {
      deallocate();
      alloc_ = other.get_allocator();
    }

    assign(other);
    return *this;
  }
  array& operator=(array&& other) {
    if (std::allocator_traits<allocator_type>::propagate_on_container_move_assignment::value) {
      swap(other);
      other.clear();
    } else {
      assign(other);
    }
    return *this;
  }

  void assign(const array& copy) {
    if (this == &copy) return;
    reallocate(copy.shape());
    construct(copy);
  }
  void assign(array&& move) {
    if (this == &move) return;
    reallocate(move.shape());
    construct(move);
  }
  void assign(Shape shape, const T& value) {
    reallocate(std::move(shape));
    construct(value);
  }

  Alloc get_allocator() const { return alloc_; }

  template <typename... Indices>
  reference at(const std::tuple<Indices...>& indices) {
    return base_[shape_.at(indices)];
  }
  template <typename... Indices>
  reference at(Indices... indices) {
    return base_[shape_.at(std::forward<Indices>(indices)...)];
  }
  template <typename... Indices>
  const_reference at(const std::tuple<Indices...>& indices) const {
    return base_[shape_.at(indices)];
  }
  template <typename... Indices>
  const_reference at(Indices... indices) const {
    return base_[shape_.at(std::forward<Indices>(indices)...)];
  }

  template <typename... Indices>
  reference operator() (const std::tuple<Indices...>& indices) {
    return base_[shape_(indices)];
  }
  template <typename... Indices>
  reference operator() (Indices... indices) {
    return base_[shape_(std::forward<Indices>(indices)...)];
  }
  template <typename... Indices>
  const_reference operator() (const std::tuple<Indices...>& indices) const {
    return base_[shape_(indices)];
  }
  template <typename... Indices>
  const_reference operator() (Indices... indices) const { 
    return base_[shape_(std::forward<Indices>(indices)...)];
  }

  template <typename Fn>
  void for_each_value(const Fn& fn) {
    for_each_index(shape(), [&](const index_type& index) {
      fn(base_[shape_(index)]);
    });
  }
  template <typename Fn>
  void for_each_value(const Fn& fn) const {
    for_each_index(shape(), [&](const index_type& index) {
      fn(base_[shape_(index)]);
    });
  }

  pointer data() { return base_; }
  const_pointer data() const { return base_; }

  const Shape& shape() const { return shape_; }
  size_type size() { return shape_.flat_extent(); }
  bool empty() const { return shape_.empty(); }
  void clear() {
    destroy();
    deallocate();
    shape_ = Shape();
  }

  /** Reshape the array. */
  void reshape(Shape new_shape) {
    assert(shape().flat_extent() == new_shape.flat_extent());
    shape_ = std::move(new_shape);
  }

  /** Compare the contents of this array to the other array. */
  bool operator!=(const array& other) {
    if (shape() != other.shape()) {
      return true;
    }

    assert(false);
    return false;
  }
  bool operator==(const array& other) { return !operator!=(other); }

  void swap(array& other) {
    using std::swap;

    if (std::allocator_traits<Alloc>::propagate_on_container_swap::value) {
      swap(alloc_, other.alloc_);
    }
    swap(base_, other.base_);
    swap(shape_, other.shape_);
  }
};

template <typename T, typename Shape, typename Alloc>
void swap(array<T, Shape, Alloc>& a, array<T, Shape, Alloc>& b) {
  a.swap(b);
}

template <typename T, typename Shape>
array<T, Shape> make_array(const Shape& shape) {
  return array<T, Shape>(shape);
}

template <typename T, typename... Extents>
auto make_dense_array(Extents... extents) {
  auto shape = make_dense_shape(std::forward<Extents>(extents)...);
  return make_array<T>(shape);
}

}  // namespace array

#endif

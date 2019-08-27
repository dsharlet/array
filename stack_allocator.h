#ifndef ARRAY_ALLOCATOR_H
#define ARRAY_ALLOCATOR_H

#include <memory>
#include <iostream>

namespace array {

/** Allocator that owns a buffer of fixed size, which will be placed
 * on the stack if the owning container is allocated on the
 * stack. This can only be used with containers that have a maximum of
 * one live allocation, which is the case for array::array. */
template <class T, std::size_t N>
class stack_allocator {
  T alloc[N];
  bool allocated;

 public:  
  typedef T value_type;

  typedef std::false_type propagate_on_container_copy_assignment;
  typedef std::false_type propagate_on_container_move_assignment;
  typedef std::false_type propagate_on_container_swap;

  stack_allocator() : allocated(false) {}
  template <class U, std::size_t U_N> constexpr 
  stack_allocator(const stack_allocator<U, U_N>&) noexcept : allocated(false) {}
  stack_allocator(const stack_allocator&) noexcept : allocated(false) {}
  stack_allocator(stack_allocator&&) noexcept : allocated(false) {}
  stack_allocator& operator=(const stack_allocator&) { return *this; }
  stack_allocator& operator=(stack_allocator&&) { return *this; }

  T* allocate(std::size_t n) {
    if (allocated) throw std::bad_alloc();
    if (n > N) throw std::bad_alloc();
    allocated = true;
    return &alloc[0];
  }
  void deallocate(T* p, std::size_t) noexcept {
    allocated = false;
  }

  static stack_allocator select_on_container_copy_construction(const stack_allocator& a) {
    return stack_allocator();
  }
};

template <class T, std::size_t T_N, class U, std::size_t U_N>
bool operator==(const stack_allocator<T, T_N>& a, const stack_allocator<U, U_N>& b) { 
  return reinterpret_cast<const void*>(&a) == reinterpret_cast<const void*>(&b);
}

template <class T, std::size_t T_N, class U, std::size_t U_N>
bool operator!=(const stack_allocator<T, T_N>& a, const stack_allocator<U, U_N>& b) {
  return reinterpret_cast<const void*>(&a) != reinterpret_cast<const void*>(&b);
}

}  // namespace

#endif

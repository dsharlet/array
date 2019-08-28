#include "array.h"
#include "test.h"
#include "lifetime.h"

namespace array {

TEST(stack_array) {
  typedef dense_array<int, 3, stack_allocator<int, 32>> stack_array_type;

  stack_array_type stack_array({4, 3, 2});
  for_all_indices(stack_array.shape(), [&](int x, int y, int c) {
    stack_array(x, y, c) = x;
  });

  stack_array_type copy_array(stack_array);
  for_all_indices(copy_array.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(copy_array(x, y, c), x);
  });

  stack_array_type assign_array;
  assign_array = stack_array;
  for_all_indices(assign_array.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(assign_array(x, y, c), x);
  });

  stack_array_type move_array(std::move(stack_array));
  for_all_indices(move_array.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(move_array(x, y, c), x);
  });

  stack_array_type move_assign;
  move_assign = std::move(assign_array);
  for_all_indices(move_assign.shape(), [&](int x, int y, int c) {
    ASSERT_EQ(move_assign(x, y, c), x);
  });
}

TEST(stack_array_move_constructor) {
  typedef dense_array<lifetime_counter, 3, stack_allocator<lifetime_counter, 32>> stack_array_type;

  stack_array_type stack_array({4, 3, 2});

  lifetime_counter::reset();
  stack_array_type move_array(std::move(stack_array));
  ASSERT_EQ(lifetime_counter::move_constructs, move_array.size());
}

TEST(stack_array_move_assignment) {
  typedef dense_array<lifetime_counter, 3, stack_allocator<lifetime_counter, 32>> stack_array_type;

  stack_array_type stack_array({4, 3, 2});

  lifetime_counter::reset();
  stack_array_type move_assign;
  move_assign = std::move(stack_array);
  // TODO: Is it OK that this assignment uses move constructions instead of move assignments?
  ASSERT_EQ(lifetime_counter::moves(), move_assign.size());
}

TEST(stack_array_bad_alloc) {
  typedef dense_array<lifetime_counter, 3, stack_allocator<lifetime_counter, 32>> stack_array_type;

  try {
    // This array is too big for our stack allocator.
    stack_array_type stack_array({4, 3, 5});
    ASSERT(false);
  } catch (std::bad_alloc) {
    // This is success.
  }
}

}  // namespace array

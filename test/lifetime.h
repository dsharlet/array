#ifndef ARRAY_TEST_LIFETIME_H
#define ARRAY_TEST_LIFETIME_H

namespace array {

struct lifetime_counter {
  static int default_constructs;
  static int copy_constructs;
  static int move_constructs;
  static int copy_assigns;
  static int move_assigns;
  static int destructs;

  static void reset() {
    default_constructs = 0;
    copy_constructs = 0;
    move_constructs = 0;
    copy_assigns = 0;
    move_assigns = 0;
    destructs = 0;
  }

  static int constructs() {
    return default_constructs + copy_constructs + move_constructs;
  }

  static int assigns() {
    return copy_assigns + move_assigns;
  }

  static int copies() {
    return copy_constructs + copy_assigns;
  }

  static int moves() {
    return move_constructs + move_assigns;
  }

  lifetime_counter() { default_constructs++; }
  lifetime_counter(const lifetime_counter&) { copy_constructs++; }
  lifetime_counter(lifetime_counter&&) { move_constructs++; }
  ~lifetime_counter() { destructs++; }

  lifetime_counter& operator=(const lifetime_counter&) { copy_assigns++; return *this; }
  lifetime_counter& operator=(lifetime_counter&&) { move_assigns++; return *this; }
};

}  // namespace array

#endif

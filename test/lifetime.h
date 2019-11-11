// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

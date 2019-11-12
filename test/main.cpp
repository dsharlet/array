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

#include "test.h"

#include <iostream>
#include <vector>

namespace {

bool wildcard_match(const char* p, const char* str) {
  while (*p && *str && (*p == *str || *p == '?') && *p != '*') {
    str++;
    p++;
  }

  if (!*p) {
    return *str == 0;
  } else if (*p == '*') {
    p++;
    do {
      if (wildcard_match(p, str)) {
        return true;
      }
    } while(*str++);
  }
  return !*p;
}

std::vector<std::pair<std::string, std::function<void()>>>& tests() {
  static std::vector<std::pair<std::string, std::function<void()>>> v;
  return v;
}

}  // namespace

namespace nda {

test::test(const std::string& name, std::function<void()> fn) {
  add_test(name, fn);
}

void add_test(const std::string& name, std::function<void()> fn) {
  tests().push_back(std::make_pair(name, fn));
}

}  // namespace nda

using namespace nda;

int main(int argc, const char** argv) {
  // By default, the filter matches all tests.
  const char* filter = "*";
  if (argc > 1) {
    filter = argv[1];
    std::cout << "Running tests matching '" << filter << "'..." << std::endl;
  }

  size_t passed = 0;
  size_t total = 0;
  for (auto& i : tests()) {
    if (!wildcard_match(filter, i.first.c_str())) continue;

    std::cout << i.first;
    try {
      i.second();
      std::cout << " passed" << std::endl;
      passed++;
    } catch(const std::exception& e) {
      std::cout << " failed: " << e.what() << std::endl;
    }
    total++;
  }
  std::cout << passed << " of " << total << " tests passed, " << tests().size() - total << " skipped" << std::endl;
  return passed < total ? -1 : 0;
}

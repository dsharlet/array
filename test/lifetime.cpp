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

#include "lifetime.h"

namespace nda {

size_t lifetime_counter::default_constructs = 0;
size_t lifetime_counter::copy_constructs = 0;
size_t lifetime_counter::move_constructs = 0;
size_t lifetime_counter::copy_assigns = 0;
size_t lifetime_counter::move_assigns = 0;
size_t lifetime_counter::destructs = 0;

}  // namespace nda

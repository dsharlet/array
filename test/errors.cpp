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

#include "array.h"

// These tests should each generate one error, with errors.cpp as the
// location of the error.
// TODO: Automate checking this.

namespace nda {

shape<dim<>, dim<>> s;
dense_array_ref<int, 2> ref;
dense_array<int, 3> a;

void shape_dim_bad_index() {
  auto z = s.template dim<2>();
}

void shape_z_bad() {
  // TODO: This one returns an error in array.h, but it's not too spammy.
  //auto z = s.z();
}

void shape_shape_too_many_dims() {
  shape<dim<>> s2(0, 1);
}

void shape_at_too_many_indices() {
  s(0, 1, 2);
}

void shape_at_too_few_indices() {
  s(0);
}


void array_ref_at_too_many_indices() {
  ref(0, 1, 2);
}

void array_ref_at_too_few_indices() {
  ref(0);
}


void array_at_too_many_indices() {
  a(0, 1, 2, 3);
}

void array_at_too_few_indices() {
  a(0, 1);
}


void copy_different_rank() {
  copy(a, ref);
}

}  // namespace nda

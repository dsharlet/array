#include "lifetime.h"

namespace array {

int lifetime_counter::default_constructs = 0;
int lifetime_counter::copy_constructs = 0;
int lifetime_counter::move_constructs = 0;
int lifetime_counter::copy_assigns = 0;
int lifetime_counter::move_assigns = 0;
int lifetime_counter::destructs = 0;

}  // namespace array

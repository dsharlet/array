#ifndef ARRAY_TEST_H
#define ARRAY_TEST_H

#include <cstdlib>
#include <cmath>
#include <sstream>
#include <functional>
#include <limits>

namespace array {

inline float randf() {
  return rand() / static_cast<float>(RAND_MAX);
}

// Base class of a test callback.
class test {
public:
  test(const std::string& name, std::function<void()> fn);
};

void add_test(const std::string& name, std::function<void()> fn);

// Exception thrown upon assertion failure.
class assert_failure : public std::runtime_error {
  std::string message_;

public:
 assert_failure(const std::string& check, const std::string& message)
   : std::runtime_error(message), message_(message) {}

  const std::string& message() const { return message_; }
};

// A stream class that builds a message, the destructor throws an
// assert_failure exception if the check fails.
class assert_stream {
  std::string check_;
  std::stringstream msg_;
  bool fail_;

public:
 assert_stream(bool condition, const std::string& check)
   : check_(check), fail_(!condition) {
    msg_ << check_;
  }
  ~assert_stream() throw(assert_failure) {
    if (fail_)
      throw assert_failure(check_, msg_.str());
  }

  template <typename T>
  assert_stream& operator << (const T& x) {
    if (fail_) {
      msg_ << x;
    }
    return *this;
  }
};

// Check if a and b are within epsilon of eachother, after normalization.
template <typename T>
bool roughly_equal(T a, T b, double epsilon = 1e-6) {
  return std::abs(a - b) < epsilon * std::max(std::max(std::abs(a), std::abs(b)), static_cast<T>(1));
}

// Make a new test object. The body of the test should follow this
// macro, e.g. TEST(equality) { ASSERT(1 == 1); }
#define TEST(name)                                                     \
  void test_##name##_body();                                           \
  ::array::test test_##name##_obj(#name, test_##name##_body);          \
  void test_##name##_body()

#define ASSERT(condition) assert_stream(condition, #condition)

#define ASSERT_EQ(a, b)                 \
  ASSERT(a == b)                        \
    << "\n" << #a << "=" << a            \
    << "\n" << #b << "=" << b << " "

#define ASSERT_REQ(a, b, epsilon)      \
  ASSERT(roughly_equal(a, b, epsilon)) \
    << "\n" << #a << "=" << a           \
    << "\n" << #b << "=" << b           \
    << "\nepsilon=" << epsilon << " "

}  // namespace array

#endif

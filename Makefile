CFLAGS := $(CFLAGS) -O2 -ffast-math #-I/usr/local/Cellar/llvm/6.0.1/include/c++/v1
CXXFLAGS := $(CXXFLAGS) -std=c++14 -march=native -Wall -Wno-unknown-pragmas -ferror-limit=3 -Wno-missing-braces
LDFLAGS :=

DEPS = \
	shape.h \
	array.h

TEST_SRC = $(wildcard test/*.cpp)
TEST_OBJ = $(TEST_SRC:%.cpp=obj/%.o)

obj/%.o: %.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I. -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

bin/test: $(TEST_OBJ)
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

bin/example: obj/examples/example.o
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

.PHONY: all clean test example

clean:
	rm -rf obj/* bin/*

test: bin/test bin/example
	bin/test $(FILTER)
	bin/example

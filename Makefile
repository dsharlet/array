CFLAGS := $(CFLAGS) -O2 -mavx #-I/usr/local/Cellar/llvm/6.0.1/include/c++/v1
CXXFLAGS := $(CXXFLAGS) -std=c++14 -march=native -Wall -Wno-unknown-pragmas -ferror-limit=3 -Wno-missing-braces
LDFLAGS :=

DEPS = array.h

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

bin/matrix: obj/examples/matrix.o
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

.PHONY: all clean test example matrix

clean:
	rm -rf obj/* bin/*

test: bin/test bin/example bin/matrix
	bin/test $(FILTER)
	bin/example
	bin/matrix

examples: bin/example bin/matrix
	bin/example
	bin/matrix

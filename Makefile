CFLAGS := $(CFLAGS) -O3 -ffast-math -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS = array.h image.h

TEST_SRC = $(wildcard test/*.cpp)
TEST_OBJ = $(TEST_SRC:%.cpp=obj/%.o)

obj/%.o: %.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I. -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

bin/test: $(TEST_OBJ)
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

bin/%: obj/examples/%.o
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

.PHONY: all clean test matrix

clean:
	rm -rf obj/* bin/*

test: bin/test bin/matrix
	bin/test $(FILTER)
	bin/matrix

examples: bin/matrix
	bin/matrix

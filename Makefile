CFLAGS := $(CFLAGS) -O2 -ffast-math -mavx
CXXFLAGS := $(CXXFLAGS) -std=c++14 -march=native -Wall -ferror-limit=3
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

bin/%: obj/examples/%.o
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

.PHONY: all clean test blur matrix

clean:
	rm -rf obj/* bin/*

test: bin/test bin/blur bin/matrix
	bin/test $(FILTER)
	bin/blur
	bin/matrix

examples: bin/blur bin/matrix
	bin/blur
	bin/matrix

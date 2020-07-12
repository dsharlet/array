CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS := array.h image.h matrix.h

TEST_SRC := $(filter-out test/errors.cpp, $(wildcard test/*.cpp))
TEST_OBJ := $(TEST_SRC:%.cpp=obj/%.o)

obj/%.o: %.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I. -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

bin/test: $(TEST_OBJ)
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/test
	bin/test $(FILTER)
	@! $(CXX) -I. -c test/errors.cpp -std=c++14 -Wall -ferror-limit=0 2>&1 | grep "error:" | grep array.h

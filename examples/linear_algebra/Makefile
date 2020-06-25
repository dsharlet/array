CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS = ../../array.h ../../image.h ../benchmark.h

bin/matrix: matrix.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ matrix.cpp $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/matrix
	bin/matrix

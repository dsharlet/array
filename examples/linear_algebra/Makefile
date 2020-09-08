CFLAGS := $(CFLAGS) -O2 -march=native -ffast-math -fstrict-aliasing -fno-exceptions -DNDEBUG
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS := ../../array.h ../../matrix.h ../benchmark.h ../../ein_reduce.h

bin/%: %.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ $< $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/matrix bin/conv2d_relu
	bin/matrix
	bin/conv2d_relu

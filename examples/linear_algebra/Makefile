CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS := ../../array.h ../../matrix.h ../benchmark.h einsum.h

bin/%: %.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ $< $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/matrix bin/conv2d bin/conv3d bin/einsum
	bin/matrix
	bin/conv2d
	bin/conv3d
	bin/einsum

CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

ARRAY_DEPS = ../../array.h ../../image.h ../benchmark.h

bin/%: %.cpp $(ARRAY_DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ $< $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/matrix bin/conv
	bin/matrix
	bin/conv

CFLAGS := $(CFLAGS) -O2 -march=native -ffast-math -fstrict-aliasing -fno-exceptions -DNDEBUG
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

ifneq ($(BLAS), )
CFLAGS += -DBLAS
LDFLAGS += -lblas
endif

DEPS := ../../include/array/array.h ../../include/array/matrix.h ../benchmark.h ../../include/array/ein_reduce.h ../../include/array/z_order.h

bin/%: %.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../include -I../ -o $@ $< $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm $(LDFLAGS)

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/matrix bin/conv2d_relu
	bin/matrix
	bin/conv2d_relu

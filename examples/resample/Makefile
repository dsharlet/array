CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS = ../../array.h ../../image.h ../benchmark.h

GRAPHICSMAGICK_CONFIG=`GraphicsMagick++-config --cppflags --cxxflags --ldflags --libs`

bin/resample: resample.cpp resample.h rational.h $(ARRAY_DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ resample.cpp $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm $(GRAPHICSMAGICK_CONFIG)

bin/benchmark: benchmark.cpp resample.h rational.h $(ARRAY_DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ benchmark.cpp $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm

.PHONY: all clean benchmark test

clean:
	rm -rf obj/* bin/* test_outputs

benchmark: bin/benchmark
	bin/benchmark 400 300 1200 900
	bin/benchmark 1200 900 400 300

test: bin/resample
	mkdir -p test_outputs/upsample
	mkdir -p test_outputs/downsample
	bin/resample test_inputs/small.png 800 600 box test_outputs/upsample/box.png
	bin/resample test_inputs/small.png 800 600 linear test_outputs/upsample/linear.png
	bin/resample test_inputs/small.png 800 600 quadratic test_outputs/upsample/quadratic.png
	bin/resample test_inputs/small.png 800 600 cubic test_outputs/upsample/cubic.png
	bin/resample test_inputs/small.png 800 600 lanczos test_outputs/upsample/lanczos.png
	bin/resample test_inputs/big.png 400 300 box test_outputs/downsample/box.png
	bin/resample test_inputs/big.png 400 300 linear test_outputs/downsample/linear.png
	bin/resample test_inputs/big.png 400 300 quadratic test_outputs/downsample/quadratic.png
	bin/resample test_inputs/big.png 400 300 cubic test_outputs/downsample/cubic.png
	bin/resample test_inputs/big.png 400 300 lanczos test_outputs/downsample/lanczos.png
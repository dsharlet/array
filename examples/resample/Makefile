CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

ARRAY_DEPS := ../../array.h ../../image.h ../benchmark.h
HEADERS := resample.h rational.h

GRAPHICSMAGICK_CONFIG := `GraphicsMagick++-config --cppflags --cxxflags --ldflags --libs`

bin/resample: resample.cpp $(HEADERS) $(ARRAY_DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ resample.cpp $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm $(GRAPHICSMAGICK_CONFIG)

bin/benchmark: benchmark.cpp $(HEADERS) $(ARRAY_DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ benchmark.cpp $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm

.PHONY: all clean benchmark test

clean:
	rm -rf obj/* bin/* test_outputs

benchmark: bin/benchmark
	bin/benchmark 400 300 1200 900
	bin/benchmark 1200 900 400 300

test: bin/resample
	mkdir -p test_outputs
	bin/resample test_inputs/small.png 800 600 box test_outputs/upsample_box.png
	bin/resample test_inputs/small.png 800 600 linear test_outputs/upsample_linear.png
	bin/resample test_inputs/small.png 800 600 quadratic test_outputs/upsample_quadratic.png
	bin/resample test_inputs/small.png 800 600 cubic test_outputs/upsample_cubic.png
	bin/resample test_inputs/small.png 800 600 lanczos test_outputs/upsample_lanczos.png
	bin/resample test_inputs/big.png 400 300 box test_outputs/downsample_box.png
	bin/resample test_inputs/big.png 400 300 linear test_outputs/downsample_linear.png
	bin/resample test_inputs/big.png 400 300 quadratic test_outputs/downsample_quadratic.png
	bin/resample test_inputs/big.png 400 300 cubic test_outputs/downsample_cubic.png
	bin/resample test_inputs/big.png 400 300 lanczos test_outputs/downsample_lanczos.png

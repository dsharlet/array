CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS = ../../array.h ../../image.h ../benchmark.h

GRAPHICSMAGICK_CONFIG=`GraphicsMagick++-config --cppflags --cxxflags --ldflags --libs`

bin/resize: resize.cpp rational.h $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I../../ -I../ -o $@ resize.cpp $(CFLAGS) $(CXXFLAGS) -lstdc++ -lm $(GRAPHICSMAGICK_CONFIG)

.PHONY: all clean test

clean:
	rm -rf obj/* bin/* test_outputs

test: bin/resize
	mkdir -p test_outputs/upsample
	mkdir -p test_outputs/downsample
	bin/resize test_inputs/small.png 800 600 nearest test_outputs/upsample/nearest.png test_outputs/upsample/nearest_magick.png
	bin/resize test_inputs/small.png 800 600 linear test_outputs/upsample/linear.png test_outputs/upsample/linear_magick.png
	bin/resize test_inputs/small.png 800 600 quadratic test_outputs/upsample/quadratic.png test_outputs/upsample/quadratic_magick.png
	bin/resize test_inputs/small.png 800 600 catmullrom test_outputs/upsample/catmullrom.png test_outputs/upsample/catmullrom_magick.png
	bin/resize test_inputs/small.png 800 600 lanczos test_outputs/upsample/lanczos.png test_outputs/upsample/lanczos_magick.png
	bin/resize test_inputs/big.png 400 300 nearest test_outputs/downsample/nearest.png test_outputs/downsample/nearest_magick.png
	bin/resize test_inputs/big.png 400 300 linear test_outputs/downsample/linear.png test_outputs/downsample/linear_magick.png
	bin/resize test_inputs/big.png 400 300 quadratic test_outputs/downsample/quadratic.png test_outputs/downsample/quadratic_magick.png
	bin/resize test_inputs/big.png 400 300 catmullrom test_outputs/downsample/catmullrom.png test_outputs/downsample/catmullrom_magick.png
	bin/resize test_inputs/big.png 400 300 lanczos test_outputs/downsample/lanczos.png test_outputs/downsample/lanczos_magick.png

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
	mkdir -p test_outputs
	bin/resize test_inputs/small.png 800 600 nearest test_outputs/nearest.png test_outputs/magick_nearest.png
	bin/resize test_inputs/small.png 800 600 linear test_outputs/linear.png test_outputs/magick_linear.png
	bin/resize test_inputs/small.png 800 600 quadratic test_outputs/quadratic.png test_outputs/magick_quadratic.png
	bin/resize test_inputs/small.png 800 600 catmullrom test_outputs/catmullrom.png test_outputs/magick_catmullrom.png
	bin/resize test_inputs/small.png 800 600 lanczos2 test_outputs/lanczos3.png test_outputs/magick_lanczos.png

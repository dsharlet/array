CFLAGS := $(CFLAGS) -O2 -ffast-math -fstrict-aliasing -march=native
CXXFLAGS := $(CXXFLAGS) -std=c++14 -Wall
LDFLAGS := $(LDFLAGS)

DEPS := array.h ein_reduce.h image.h matrix.h

TEST_SRC := $(filter-out test/errors.cpp, $(wildcard test/*.cpp))
TEST_OBJ := $(TEST_SRC:%.cpp=obj/%.o)

# Note that .cu files are automatically compiled by clang as CUDA.
# Other extensions will need "-x cuda".
CUDA_TEST_SRC := $(wildcard test/*.cu)

obj/%.o: %.cpp $(DEPS)
	mkdir -p $(@D)
	$(CXX) -I. -c -o $@ $< $(CFLAGS) $(CXXFLAGS)

bin/test: $(TEST_OBJ)
	mkdir -p $(@D)
	$(CXX) -o $@ $^ $(LDFLAGS) -lstdc++ -lm

cuda_build_test: $(CUDA_TEST_SRC) $(DEPS)
	$(CXX) -I. -c $< $(CFLAGS) $(CXXFLAGS) --cuda-gpu-arch=sm_52 -nocudalib -nocudainc -emit-llvm
	# TODO: Figure out how to run this build test to produce outputs in bin/ or obj/
	rm *.bc

.PHONY: all clean test

clean:
	rm -rf obj/* bin/*

test: bin/test
	@! $(CXX) -I. -c test/errors.cpp -std=c++14 -Wall -ferror-limit=0 2>&1 | grep "error:" | grep array.h
	bin/test $(FILTER)

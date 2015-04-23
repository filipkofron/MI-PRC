CUDA_FILES := $(wildcard *.cu)
OBJ_FILES := $(addprefix ,$(notdir $(CUDA_FILES:.cu=.o)))
REQ_FILES := $(wildcard *.cuh)
CUDA_FLAGS := -maxrregcount 24 -Xptxas -v -gencode=arch=compute_20,code=compute_20 --relocatable-device-code true -O3
LD_FLAGS := -std=c++11 -lcudadevrt -lcuda $(CUDA_FLAGS)
CC_FLAGS := -std=c++11 $(CUDA_FLAGS)

raytracer.run: $(OBJ_FILES)
	nvcc $(LD_FLAGS) -o $@ $^

%.o: %.cu $(REQ_FILES)
	nvcc $(CC_FLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f raytracer.run

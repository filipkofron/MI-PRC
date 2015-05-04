CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(addprefix ,$(notdir $(CPP_FILES:.cpp=.o)))
REQ_FILES := $(wildcard *.h)
CUDA_FLAGS := -O3
LD_FLAGS := -std=c++11 $(CUDA_FLAGS)
CC_FLAGS := -std=c++11 $(CUDA_FLAGS)

raytracer.run: $(OBJ_FILES)
	clang++ $(LD_FLAGS) -o $@ $^ -pthread -std=c++11

%.o: %.cpp $(REQ_FILES)
	clang++ $(CC_FLAGS) -c -o $@ $< -std=c++11

clean:
	rm -f *.o
	rm -f raytracer.run

# Compilers
CC = g++
NVCC = nvcc

# Flags
CFLAGS = -std=c++17 -O2
NVFLAGS = -O2

# Sources
SRCS_CPP = timePerlin.cpp
SRCS_CU  = perlin.cu

OBJS_CPP = $(SRCS_CPP:.cpp=.o)
OBJS_CU  = $(SRCS_CU:.cu=.o)
OBJS = $(OBJS_CPP) $(OBJS_CU)

EXEC = perlin

# Build rules
$(EXEC): $(OBJS)
	$(NVCC) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

test: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(OBJS) $(EXEC)


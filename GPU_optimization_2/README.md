## GPU Optimization 1

### Problem 

In the GPU_basic directory, we give each thread a pretty hefty load:
1. First, it must generate four pseudo-random gradients using trigonometric functions.
2. Next, it does a sequential series of linear algebra and mathematical operations using those generated gradients
3. Finally, it writes the value to global memory

In this directory, we will focus on tackling the first problem, which is the gradient generation. Rather than having every pixel generate the gradients, we will precompute them and store them in the GPU's texture memory. We could (and should) come up with a different means for generating pseudo-random data that doesn't involve two trigonometric functions. 
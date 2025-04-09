
// CUDA kernel - each thread computes the Perlin noise value for a single pixel
__global__ void perlinKernel(float* d_image, int windowWidth, int windowHeight, int gridSize, int numOctaves, unsigned seed) {
    /** 
     * 1. Find out which pixel this thread is working on via thread id, block idx, etc.
     * 2. Find the four corners that box in the given point
     * 3. Generate the random vectors for those points (or load them from memory?)
     * 4. Compute the distance between pixel and each of the points
     * 5. Compute dot product between each distance vector and the random vector of its corresponding corner
     * 6. Do interpolation
     * 7. (maybe?) map to a color value?
     *
     * NOTES:
     *  - __device__ can be used to define a function that gets executed only on the device
     *  - 
     */
}

// Builds the perlin noise map 
void buildPerlinNoise(int windowWidth, int windowHeight, int gridSize, int numOctaves, unsigned seed, float** outImage) {
    // Initialize host + device memory
    float* h_image;
    float* d_image;

    // Amt of data to copy back and forth
    int bytes = windowWidth * windowHeight * sizeof(float);

    // Allocate host + device memory
    h_image = (float*)malloc(bytes)
    cudaMalloc(&d_image, bytes);

    // device kernel launch config
    dim3 blockDim(16, 16);
    dim3 gridDim((windowWidth + 15) / 16, (windowHeight + 15) / 16);

    // Launch device kernel
    perlinKernel<<<gridDim, blockDim>>>(d_image, windowWidth, windowHeight, gridSize, numOctaves, seed);

    // Bring the data back to the host
    cudaMemcpy(h_image, d_image, windowWidth * windowHeight * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert 1D to 2D
    *outImage = new float[windowWidth * windowHeight];
    memcpy(*outImage, h_image, windowWidth * windowHeight * sizeof(float));

    // Cleanup
    cudaFree(d_image);
    free (h_image);
}
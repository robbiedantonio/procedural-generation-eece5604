#include "perlin.hpp"

// CUDA-compatible 2D vector struct
struct vector2 {
    float x, y;
};

// Timing function
double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

// Generate a pseudo-random gradient vector based on the given coordinates and seed
__device__ vector2 randomGradient(int ix, int iy, unsigned seed) {
    unsigned a = ix + seed;
    unsigned b = iy + seed;

    a *= 3284157443U;
    b ^= a << 21 | a >> 11;
    b *= 1911520717U;
    a ^= b << 4 | b >> 28;
    a *= 2048419325U;

    float random = a * (3.14159265f / ~(~0u >> 1)); // in [0, 2π]

    vector2 v;
    v.x = sinf(random);
    v.y = cosf(random);

    return v;
}

// Compute the dot product of the gradient vector and the distance vector from the grid point to the pixel
__device__ float dotGridGradient(int ix, int iy, float x, float y, unsigned seed) {
    vector2 gradient = randomGradient(ix, iy, seed);

    float dx = x - (float)ix;
    float dy = y - (float)iy;

    return (dx * gradient.x + dy * gradient.y);
}

// Interpolate between two values using smoothstep interpolation
__device__ float interpolate(float a0, float a1, float w) {
    return (a1 - a0) * (3.0f - w * 2.0f) * w * w + a0;
}

// Compute the Perlin noise value at a given point (x, y) using the specified seed
__device__ float pixelPerlin(float x, float y, unsigned seed) {
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = x - (float)x0;
    float sy = y - (float)y0;

    float n0 = dotGridGradient(x0, y0, x, y, seed);
    float n1 = dotGridGradient(x1, y0, x, y, seed);
    float ix0 = interpolate(n0, n1, sx);

    n0 = dotGridGradient(x0, y1, x, y, seed);
    n1 = dotGridGradient(x1, y1, x, y, seed);
    float ix1 = interpolate(n0, n1, sx);

    return interpolate(ix0, ix1, sy);
}

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
    // Get the pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within bounds
    if (x >= windowWidth || y >= windowHeight) return;

    // Initialize the noise, amplitude, and frequency
    float noise = 0.0f;
    float freq = 1.0f;
    float amp = 1.0f;

    // Loop through the octaves to accumulate the noise value
    for (int i = 0; i < numOctaves; i++) {
        noise += amp * pixelPerlin(x * freq / gridSize, y * freq / gridSize, seed);
        freq *= 2.0f; // Increase frequency for the next octave
        amp /= 2.0f; // Decrease amplitude for the next octave
    }

    // Normalize the noise value to the range [0, 1]
    noise = fminf(1.0f, fmaxf(-1.0f, noise)); // Clamp the value to [-1, 1]

    d_image[y * windowWidth + x] = noise; // Write the noise to the global array
}

// Builds the perlin noise map 
double buildPerlinNoise(int windowWidth, int windowHeight, int gridSize, int numOctaves, unsigned seed, float** outImage) {
    // Timing variables
    double start, finish;
    
    // Amt of data to copy back and forth
    int bytes = windowWidth * windowHeight * sizeof(float);

    // Initialize and allocate host + device memory
    float* h_image = (float*)malloc(bytes);
    float* d_image;
    cudaMalloc(&d_image, bytes);

    // device kernel launch config
    dim3 blockDim(16, 16);
    dim3 gridDim((windowWidth + 15) / 16, (windowHeight + 15) / 16);

    // Launch device kernel
    start = CLOCK();
    perlinKernel<<<gridDim, blockDim>>>(d_image, windowWidth, windowHeight, gridSize, numOctaves, seed);

    // Bring the data back to the host
    cudaMemcpy(h_image, d_image, windowWidth * windowHeight * sizeof(float), cudaMemcpyDeviceToHost);
    finish = CLOCK();
    
    // Convert 1D to 2D
    *outImage = new float[windowWidth * windowHeight];
    memcpy(*outImage, h_image, windowWidth * windowHeight * sizeof(float));

    // Cleanup
    cudaFree(d_image);
    free (h_image);

    return (finish - start);
}
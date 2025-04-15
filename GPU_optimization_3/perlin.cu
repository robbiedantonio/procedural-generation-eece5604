#include "perlin.hpp"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

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
__global__ void perlinKernel(float* d_output, int windowWidth, int windowHeight, 
    int perlinGridSize, int numOctaves, unsigned baseSeed) {
    // Get the pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within bounds
    bool pixelInBounds = (x < windowWidth && y < windowHeight);

    // Shared memory to hold octave results for this block
    __shared__ float octaveResults[BLOCK_SIZE_Y][BLOCK_SIZE_X][12]; // Assuming 12 octaves max

    // Initialize shared memory
    if (pixelInBounds) {
        // Each thread in the block will compute a specific octave for its pixel
        // We'll do this using a loop since we typically have fewer threads than octaves
        for (int octave = 0; octave < numOctaves; octave += blockDim.z) {
            int currentOctave = octave + threadIdx.z;

            if (currentOctave < numOctaves) {
                // Calculate frequency and amplitude for this octave
                float freq = powf(2.0f, currentOctave);
                float amp = powf(0.5f, currentOctave);

                // Compute noise for this octave
                float noise = pixelPerlin(x * freq / perlinGridSize, 
                    y * freq / perlinGridSize,
                    baseSeed + currentOctave);

                // Normalize and store in shared memory
                noise = fminf(1.0f, fmaxf(-1.0f, noise));
                octaveResults[threadIdx.y][threadIdx.x][currentOctave] = amp * noise;
            } else {
                octaveResults[threadIdx.y][threadIdx.x][currentOctave] = 0.0f;
            }
        }
    }

    // Ensure all octaves are computed before summing
    __syncthreads();

    // Sum up the octaves and write to global memory
    if (pixelInBounds) {
        float total = 0.0f;
        for (int i = 0; i < numOctaves; i++) {
            total += octaveResults[threadIdx.y][threadIdx.x][i];
        }

        // Write final result to global memory
        d_output[y * windowWidth + x] = total;
    }
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
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, 12); // 12 octaves max
    dim3 gridDim((windowWidth + blockDim.x - 1) / blockDim.x,
    (windowHeight + blockDim.y - 1) / blockDim.y,
    1);
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
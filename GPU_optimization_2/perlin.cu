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


// CUDA kernel - each thread computes the Perlin noise value for a single pixel for one octave
__global__ void perlinKernel(float* d_output, int windowWidth, int windowHeight, int perlinGridSize, float freq, float amp, unsigned seed) {
    // Get the pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within bounds
    if (x >= windowWidth || y >= windowHeight) return;

    // Compute the Perlin noise value at the given pixel coordinates
    float noise = pixelPerlin(x * freq / perlinGridSize, y * freq / perlinGridSize, seed);

    // Normalize the noise value to the range [0, 1]
    noise = fminf(1.0f, fmaxf(-1.0f, noise)); // Clamp the value to [-1, 1]

    d_output[y * windowWidth + x] += amp * noise; // Add the noise to the global array
}

// CUDA kernel - each thread adds the octaves for one pixel
__global__ void accumulateOctavesKernel(float* d_finalHeightmap, float** d_octaves, int numOctaves, int windowWidth, int windowHeight) {
    // Get the pixel coordinates for this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the pixel is within bounds
    if (x >= windowWidth || y >= windowHeight) return;

    // Calculate 1D index for the current pixel
    int idx = y * windowWidth + x;
    
    // Accumulate noise from all octaves
    float total = 0.0f;
    
    for (int i = 0; i < numOctaves; i++) {
        total += d_octaves[i][idx];
    }
    
    // Store the final accumulated noise value
    d_finalHeightmap[idx] = total;
}


// Builds the perlin noise map 
double buildPerlinNoise(int windowWidth, int windowHeight, int perlinGridSize, int numOctaves, unsigned seed, float** outHeightmap) {
    // Timing variables
    double start, finish;
    
    // Amt of data per octave
    int heightmapBytes = windowWidth * windowHeight * sizeof(float);
    int octaveBytes = numOctaves * sizeof(float*);

    // Initialize and allocate heightmap in host + device memory
    float* h_heightmap = (float*)malloc(heightmapBytes);
    memset(h_heightmap, 0, heightmapBytes); // Initialize the height map to 0
    
    float* d_heightmap;
    cudaMalloc(&d_heightmap, heightmapBytes);
    cudaMemset(d_heightmap, 0, heightmapBytes); // Initialize device memory to 0

    // Initialize host octave pointers
    float** h_octaves = (float**)malloc(octaveBytes);

    // Initialize device octave pointers
    float** d_octaves;
    cudaMalloc(&d_octaves, octaveBytes);

    // Allocate memory for each octave
    for (int i = 0; i < numOctaves; i++) {
        float* d_octave;
        cudaMalloc(&d_octave, heightmapBytes);
        cudaMemset(d_octave, 0, heightmapBytes); // Initialize to 0
        h_octaves[i] = d_octave; // Store the device pointers in the host array
    }

    // Copy the array of pointers to the device
    cudaMemcpy(d_octaves, h_octaves, octaveBytes, cudaMemcpyHostToDevice);

    // device kernel launch config
    dim3 blockDim(16, 16);
    dim3 gridDim((windowWidth + blockDim.x - 1) / blockDim.x, (windowHeight + blockDim.y - 1) / blockDim.y);

    // Starting amplitude and frequency
    float frequency = 1.0f;
    float amplitude = 1.0f;

    // Declare and create CUDA streams
    cudaStream_t* streams = (cudaStream_t*)malloc(numOctaves * sizeof(cudaStream_t));
    for (int i = 0; i < numOctaves; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch device kernels as streams so that they can be executed in parallel
    start = CLOCK();

    // Call the octave kernels as a stream so that they can be executed in parallel
    for (int i = 0; i < numOctaves; i++) {
        perlinKernel<<<gridDim, blockDim, 0, streams[i]>>>(
            h_octaves[i], windowWidth, windowHeight, 
            perlinGridSize, frequency, amplitude, 
            seed + i); // Use a different seed for each octave
        
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }

    // Synchronize the streams
    for (int i = 0; i < numOctaves; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    // Launch the octave accumulation kernel
    accumulateOctavesKernel<<<gridDim, blockDim>>>(
        d_heightmap, d_octaves, numOctaves, 
        windowWidth, windowHeight);
    
    // Wait for the accumulation to complete
    // cudaDeviceSynchronize();

    // Bring the data back to the host
    cudaMemcpy(h_heightmap, d_heightmap, heightmapBytes, cudaMemcpyDeviceToHost);
    finish = CLOCK();
    
    // Set the output heightmap
    *outHeightmap = (float*)malloc(heightmapBytes);
    memcpy(*outHeightmap, h_heightmap, heightmapBytes);

    // Cleanup
    cudaFree(d_heightmap);
    for (int i = 0; i < numOctaves; i++) {
        cudaFree(h_octaves[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_octaves);
    free(h_octaves);
    free(streams);
    free(h_heightmap);

    return (finish - start);
}
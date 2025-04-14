#include <math_constants.h>

// Timing function
double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}



// CUDA kernel to precompute the grid gratients
__global__ void generateGridGradients(float4* gradients, int perlinGridSize, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < perlinGridSize && y < perlinGridSize) {
        // Create a pseudo-random hash from the index and seed
        unsigned int idx = y * perlinGridSize + x;
        unsigned int hash = seed;
        hash ^= x * 1523 + y * 823;
        hash *= 0x85ebca6b;
        hash ^= hash >> 13;
        hash *= 0xc2b2ae35;
        hash ^= hash >> 16;
        
        // Generate a unit vector in a random direction
        float angle = hash * (2.0f * CUDART_PI_F / UINT_MAX);
        
        // Store as normalized vector components
        gradients[idx].x = __sinf(angle);   // CUDA sin/cos intrinsics
        gradients[idx].y = __cosf(angle);
        gradients[idx].z = 0.0f;
    }
}

// Interpolate between two values using smoothstep interpolation
__device__ float interpolate(float a0, float a1, float w) {
    return (a1 - a0) * (3.0f - w * 2.0f) * w * w + a0;
}

// Compute the Perlin noise value at a given point (x, y) using the specified seed
__device__ float pixelPerlin(float x, float y, cudaTextureObject_t gradientTexObj) {
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = x - (float)x0;
    float sy = y - (float)y0;

    // Convert to normalized coordinates for the texture
    float u0 = (float)x0;// / perlinGridSize;
    float v0 = (float)y0;// / perlinGridSize;
    float u1 = (float)x1;// / perlinGridSize;
    float v1 = (float)y1;// / perlinGridSize;

    // Read the gradient vectors from the texture memory using the texture object
    float4 g00 = tex2D<float4>(gradientTexObj, u0, v0);
    float4 g10 = tex2D<float4>(gradientTexObj, u1, v0);
    float4 g01 = tex2D<float4>(gradientTexObj, u0, v1);
    float4 g11 = tex2D<float4>(gradientTexObj, u1, v1);

    // Compute the dot products between the gradients and the distance vectors
    float dx0 = x - (float)x0;
    float dy0 = y - (float)y0;
    float dx1 = x - (float)x1;
    float dy1 = y - (float)y1;

    float n00 = g00.x * dx0 + g00.y * dy0;
    float n10 = g10.x * dx1 + g10.y * dy0;
    float n01 = g01.x * dx0 + g01.y * dy1;
    float n11 = g11.x * dx1 + g11.y * dy1;

    // Interpolate along x
    float ix0 = interpolate(n00, n10, sx);
    float ix1 = interpolate(n01, n11, sx);

    // Interpolate along y
    return interpolate(ix0, ix1, sy);
}

// CUDA kernel to generate Perlin noise 
__global__ void perlinKernel(float* d_image, int windowWidth, int windowHeight, int perlinGridSize, int numOctaves, unsigned seed, cudaTextureObject_t gradientTexObj) {
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
        noise += amp * pixelPerlin(x * freq / perlinGridSize, y * freq / perlinGridSize, gradientTexObj);
        freq *= 2.0f; // Increase frequency for the next octave
        amp /= 2.0f; // Decrease amplitude for the next octave
    }

    // Normalize the noise value to the range [0, 1]
    noise = fminf(1.0f, fmaxf(-1.0f, noise)); // Clamp the value to [-1, 1]

    d_image[y * windowWidth + x] = noise; // Write the noise to the global array
}

// Build the Perlin Noise heightmap
double buildPerlinNoise(int windowWidth, int windowHeight, int perlinGridSize, int numOctaves, unsigned seed, float** outImage) {
    // Timing
    double start, finish;
    
    /** Generate gradients for perlin noise */
    // Allocate device memory
    float4 *d_gradients;
    cudaMalloc(&d_gradients, perlinGridSize * perlinGridSize * sizeof(float4));
    
    // Device kernel configuration
    dim3 blockSize(16, 16);
    dim3 gradientGridDim((perlinGridSize + blockSize.x - 1) / blockSize.x,
                 (perlinGridSize + blockSize.y - 1) / blockSize.y);
    
    // Call device kernel
    start = CLOCK();
    generateGridGradients<<<gradientGridDim, blockSize>>>(d_gradients, perlinGridSize, seed);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Move result to texture memory:
    
    // 1. Create a CUDA array for the texture data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray* gradientArray;
    cudaMallocArray(&gradientArray, &channelDesc, perlinGridSize, perlinGridSize);
    
    // 2. Copy the gradient data from device memory to the CUDA array
    cudaMemcpy2DToArray(gradientArray, 0, 0, 
                        d_gradients, perlinGridSize * sizeof(float4),
                        perlinGridSize * sizeof(float4), perlinGridSize,
                        cudaMemcpyDeviceToDevice);
    
    // 3. Set up the texture resource descriptor
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = gradientArray;
    
    // 4. Set up the texture descriptor
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;  // Wrap addressing for Perlin noise
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;     // Bilinear interpolation
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;                  // Use normalized coordinates [0,1]
    
    // 5. Create the texture object
    cudaTextureObject_t gradientTexObj = 0;
    cudaCreateTextureObject(&gradientTexObj, &resDesc, &texDesc, NULL);
    
    // Remember to clean up the texture object and array when done
    // cudaDestroyTextureObject(gradientTexObj);
    // cudaFreeArray(gradientArray);
    // cudaFree(d_gradients);

    // Amt of data to copy back and forth
    int bytes = windowWidth * windowHeight * sizeof(float);

    // Initialize and allocate host + device memory
    float* h_image = (float*)malloc(bytes);
    float* d_image;
    cudaMalloc(&d_image, bytes);

    // device kernel launch config
    dim3 noiseGridDim((windowWidth + 15) / 16, (windowHeight + 15) / 16);

    // Launch device kernel
    start = CLOCK();
    perlinKernel<<<noiseGridDim, blockSize>>>(d_image, windowWidth, windowHeight, 
        perlinGridSize, numOctaves, seed, 
        gradientTexObj);

    // Bring the data back to the host
    cudaMemcpy(h_image, d_image, windowWidth * windowHeight * sizeof(float), cudaMemcpyDeviceToHost);
    finish = CLOCK();
    
    // Convert 1D to 2D
    *outImage = new float[windowWidth * windowHeight];
    memcpy(*outImage, h_image, windowWidth * windowHeight * sizeof(float));

    // Cleanup
    cudaDestroyTextureObject(gradientTexObj);
    cudaFreeArray(gradientArray);
    cudaFree(d_gradients);
    cudaFree(d_image);
    free(h_image);

    return (finish - start);
}


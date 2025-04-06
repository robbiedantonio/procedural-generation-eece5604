#include <cmath>
#include <cuda_runtime.h>


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

__device__ float dotGridGradient(int ix, int iy, float x, float y, unsigned seed) {
    vector2 gradient = randomGradient(ix, iy, seed);

    float dx = x - (float)ix;
    float dy = y - (float)iy;

    return (dx * gradient.x + dy * gradient.y);
}

__device__ float interpolate(float a0, float a1, float w) {
    return (a1 - a0) * (3.0f - w * 2.0f) * w * w + a0;
}

__device__ float perlin(float x, float y, unsigned seed) {
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

__global__ void buildImageKernel(float* image, int width, int height, int gridSize, int numOctaves, unsigned seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float val = 0.0f;
    float freq = 1.0f;
    float amp = 1.0f;

    for (int i = 0; i < numOctaves; i++) {
        val += perlin(x * freq / gridSize, y * freq / gridSize, seed) * amp;
        freq *= 2.0f;
        amp /= 2.0f;
    }

    val *= 1.2f;
    val = fminf(1.0f, fmaxf(-1.0f, val));

    image[y * width + x] = val;
}

void buildImage(int windowWidth, int windowHeight, int gridSize, int numOctaves, unsigned seed, float** outImage) {
    float* d_image;
    float* h_image = new float[windowWidth * windowHeight];

    cudaMalloc(&d_image, windowWidth * windowHeight * sizeof(float));

    dim3 blockDim(16, 16);
    dim3 gridDim((windowWidth + 15) / 16, (windowHeight + 15) / 16);

    buildImageKernel<<<gridDim, blockDim>>>(d_image, windowWidth, windowHeight, gridSize, numOctaves, seed);
    cudaMemcpy(h_image, d_image, windowWidth * windowHeight * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert 1D to 2D
    *outImage = new float[windowWidth * windowHeight];
    memcpy(*outImage, h_image, windowWidth * windowHeight * sizeof(float));

    cudaFree(d_image);
    delete[] h_image;
}

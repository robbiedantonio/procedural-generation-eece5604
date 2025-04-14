#ifndef PERLIN_HPP
#define PERLIN_HPP

// Main Perlin noise image builder (runs on GPU)
// Parameters:
// - windowWidth, windowHeight: image size
// - gridSize: base frequency for Perlin noise
// - numOctaves: number of noise layers
// - seed: random seed for gradient hashing
// - outImage: pointer to a heap-allocated 1D float array (flattened 2D image)
//             Caller is responsible for deleting this memory.
double buildPerlinNoise(int windowWidth, int windowHeight, int gridSize, int numOctaves, unsigned seed, float** outImage);

#endif // PERLIN_HPP

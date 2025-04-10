#include <iostream>
#include <math.h>
#include "perlin.hpp"

// Define a 2D vector structure
typedef struct {
    float x, y;
} vector2;

// Timing function
double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,  &t);
    return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

vector2 randomGradient(int ix, int iy, unsigned seed) {
    // Use the seed and grid coordinates to create a deterministic "random" number
    unsigned a = ix + seed;  // Combine grid coordinates with the seed
    unsigned b = iy + seed;
    
    // Bit manipulation to generate a deterministic "random" value
    a *= 3284157443;
    b ^= a << 21 | a >> 11;
    b *= 1911520717;
    a ^= b << 4 | b >> 28;
    a *= 2048419325;

    float random = a * (3.14159265 / ~(~0u >> 1)); // in [0, 2*Pi]
    
    // Create the vector from the angle
    vector2 v;
    v.x = sin(random);
    v.y = cos(random);

    return v;
}


// Computes the dot product of the distance and gradient vectors.
float dotGridGradient(int ix, int iy, float x, float y, unsigned seed) {
    // Get gradient from integer coordinates with seed
    vector2 gradient = randomGradient(ix, iy, seed);

    // Compute the distance vector
    float dx = x - (float)ix;
    float dy = y - (float)iy;

    // Compute the dot-product
    return (dx * gradient.x + dy * gradient.y);
}

float interpolate(float a0, float a1, float w)
{
    return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0;
}

// Sample Perlin noise at coordinates x, y
float pixelPerlin(float x, float y, unsigned seed) {
    
    // Determine grid cell corner coordinates
    int x0 = (int)x; 
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Compute Interpolation weights
    float sx = x - (float)x0;
    float sy = y - (float)y0;
    
    // Compute and interpolate top two corners
    float n0 = dotGridGradient(x0, y0, x, y, seed);
    float n1 = dotGridGradient(x1, y0, x, y, seed);
    float ix0 = interpolate(n0, n1, sx);

    // Compute and interpolate bottom two corners
    n0 = dotGridGradient(x0, y1, x, y, seed);
    n1 = dotGridGradient(x1, y1, x, y, seed);
    float ix1 = interpolate(n0, n1, sx);

    // Final step: interpolate between the two previously interpolated values, now in y
    float value = interpolate(ix0, ix1, sy);
    
    return value;
}

double buildPerlinNoise(const int windowWidth, const int windowHeight, const int gridSize, const int numOctaves, unsigned seed, float **image) {
    // timing variables
    double start, finish;

    start = CLOCK();

    for (int x = 0; x < windowWidth; x++) {
        for (int y = 0; y < windowHeight; y++) {
            float val = 0;  // Value of the pixel
            float freq = 1; // Frequency
            float amp = 1;  // Amplitude

            for (int i = 0; i < numOctaves; i++) {
                val += pixelPerlin(x * freq / gridSize, y * freq / gridSize, seed) * amp;   // Get perlin noise for an octave
                freq *= 2;  // Next octave has double the frequency
                amp /= 2;   // Next octave has half the amplitude
            }

            // Clamp the value to be in [-1.0, 1.0]
            val *= 1.2;
            if (val > 1.0f) val = 1.0f;
            else if (val < -1.0f) val = -1.0f;

            // Write value into the image array
            image[x][y] = val;
        }
    }

    finish = CLOCK();
    return (finish - start);
}

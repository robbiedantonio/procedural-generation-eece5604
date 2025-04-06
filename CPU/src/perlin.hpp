#ifndef PERLIN_HPP
#define PERLIN_HPP

/**
 * Build an image using perlin noise
 * 
 * Largely adopted from https://www.youtube.com/watch?v=kCIaHqb60Cw
 */

// Define a 2D vector structure
typedef struct {
    float x, y;
} vector2;

// Function to generate a random gradient vector based on grid coordinates
vector2 randomGradient(int ix, int iy);

// Function to compute the dot product of the distance and gradient vectors
float dotGridGradient(int ix, int iy, float x, float y);

// Function to interpolate between two values with a smooth curve
float interpolate(float a0, float a1, float w);

// Function to generate Perlin noise at given coordinates
float perlin(float x, float y);

// Function to build the Perlin noise map
void buildImage(const int windowWidth, const int windowHeight, const int gridSize, const int numOctaves, const unsigned seed, float **image);


#endif

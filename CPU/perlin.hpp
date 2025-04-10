#ifndef PERLIN_HPP
#define PERLIN_HPP

/**
 * Build an image using perlin noise
 * 
 * Largely adopted from https://www.youtube.com/watch?v=kCIaHqb60Cw
 */
// Function to build the Perlin noise map
double buildPerlinNoise(const int windowWidth, const int windowHeight, const int gridSize, const int numOctaves, const unsigned seed, float **image);


#endif

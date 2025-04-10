#include <iostream>
#include <chrono>
#include "perlin.hpp"  // Your CUDA-enabled buildImage() should be declared here

int main() {
    const int windowWidth = 1920;
    const int windowHeight = 1080;
    const int GRID_SIZE = 400;
    const int NUM_OCTAVES = 12;

    // Timing variables
    double time;

    // Allocate an image array
    float** image = (float**)malloc(windowWidth * sizeof(float*));
    for (int x = 0; x < windowWidth; x++) {
        image[x] = (float*)malloc(windowHeight * sizeof(float));
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    time = buildPerlinNoise(windowWidth, windowHeight, GRID_SIZE, NUM_OCTAVES, seed, image);

    printf("Runtime: %.4f ms\n", time);

    delete[] image;
    return 0;
}

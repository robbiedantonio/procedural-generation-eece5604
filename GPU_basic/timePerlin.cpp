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

    // Allocate image as flat buffer for CUDA compatibility
    float* image = nullptr;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    time = buildPerlinNoise(windowWidth, windowHeight, GRID_SIZE, NUM_OCTAVES, seed, &image);

    printf("Runtime: %.4f ms\n", time);

    delete[] image;
    return 0;
}

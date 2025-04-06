#include <iostream>
#include <chrono>
#include "perlin.hpp"  // Your CUDA-enabled buildImage() should be declared here

void timeBuildImage(int width, int height, int gridSize, int numOctaves, unsigned seed, float** image) {
    auto start = std::chrono::high_resolution_clock::now();

    buildImage(width, height, gridSize, numOctaves, seed, image);

    cudaDeviceSynchronize();  // Ensure GPU computation is finished before measuring time

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;

    std::cout << "Terrain generated in " << elapsedSeconds.count() << " seconds." << std::endl;
}

int main() {
    const int windowWidth = 1920;
    const int windowHeight = 1080;
    const int GRID_SIZE = 400;
    const int NUM_OCTAVES = 12;

    // Allocate image as flat buffer for CUDA compatibility
    float* image = nullptr;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    timeBuildImage(windowWidth, windowHeight, GRID_SIZE, NUM_OCTAVES, seed, &image);

    // You could write image to file here if desired

    delete[] image;
    return 0;
}

#include <iostream>
#include <chrono>
#include "SFML/Graphics.hpp"
#include "perlin.hpp"

// Timing wrapper function
void timeBuildImage(int width, int height, int gridSize, int numOctaves, unsigned seed, float** image) {
    auto start = std::chrono::high_resolution_clock::now();

    buildImage(width, height, gridSize, numOctaves, seed, image);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSeconds = end - start;

    std::cout << "Terrain generated in " << elapsedSeconds.count() << " seconds." << std::endl;
}

int main() {
    const int windowWidth = 1920;
    const int windowHeight = 1080;

    sf::RenderWindow window(sf::VideoMode({windowWidth, windowHeight}), "Perlin Noise");

    sf::Image displayableImage({(unsigned int)windowWidth, (unsigned int)windowHeight}, sf::Color::Black);
    
    const int GRID_SIZE = 400;
    const int NUM_OCTAVES = 12;

    // Allocate memory for the image
    float** image = (float**)malloc(windowWidth * sizeof(float*));
    for (int x = 0; x < windowWidth; x++) {
        image[x] = (float*)malloc(windowHeight * sizeof(float));
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    timeBuildImage(windowWidth, windowHeight, GRID_SIZE, NUM_OCTAVES, seed, image);

    for (int x = 0; x < windowWidth; x++) {
        for (int y = 0; y < windowHeight; y++) {
            float value = image[x][y]; // [-1.0, 1.0]
            float normalized = (value + 1.0f) * 0.5f; // [0.0, 1.0]

            sf::Color color;

            if (normalized < 0.3f) {
                color = sf::Color(0, 0, (int)(normalized * 255 + 50)); // Deep water
            } else if (normalized < 0.4f) {
                color = sf::Color(240, 220, 180); // Beach
            } else if (normalized < 0.6f) {
                color = sf::Color(20, (int)(normalized * 255), 20); // Grassland
            } else if (normalized < 0.8f) {
                color = sf::Color((int)(normalized * 200), 160, 100); // Hills
            } else {
                int gray = (int)(normalized * 255);
                color = sf::Color(gray, gray, gray); // Mountain
            }

            displayableImage.setPixel(sf::Vector2u(x, y), color);
        }
    }

    sf::Texture texture;
    if (!texture.loadFromImage(displayableImage)) {
        std::cerr << "Failed to load texture from image!" << std::endl;
        return 1;
    }

    sf::Sprite sprite(texture);

    while (window.isOpen()) {
        if (auto event = window.pollEvent()) { 
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }

    // Cleanup
    for (int x = 0; x < windowWidth; x++) {
        free(image[x]);
    }
    free(image);    

    return 0;
}

#include <iostream>
#include "SFML/Graphics.hpp"
#include "perlin.hpp"

/**
 * Build an image using perlin noise
 * 
 * Largely adopted from https://www.youtube.com/watch?v=kCIaHqb60Cw
 */
int main() {
    const int windowWidth = 1920;
    const int windowHeight = 1080;

    sf::RenderWindow window(sf::VideoMode({windowWidth, windowHeight}), "Perlin Noise");

    sf::Image image({(unsigned int)windowWidth, (unsigned int)windowHeight}, sf::Color::Black);
    
    const int GRID_SIZE = 400;

    for (int x = 0; x < windowWidth; x++) {
        for (int y = 0; y < windowHeight; y++) {
            float val = 0;
            float freq = 1;
            float amp = 1;

            for (int i = 0; i < 12; i++) {
                val += perlin(x * freq / GRID_SIZE, y * freq / GRID_SIZE) * amp;
                freq *= 2;
                amp /= 2;
            }

            val *= 1.2;
            if (val > 1.0f) val = 1.0f;
            else if (val < -1.0f) val = -1.0f;

            int color = (int)(((val + 1.0f) * 0.5f) * 255);

            image.setPixel({(unsigned int)x, (unsigned int)y}, sf::Color(color, color, color, 255));
        }
    }

    sf::Texture texture;
    texture.loadFromImage(image);
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

    return 0;
}
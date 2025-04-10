CC = g++

## Update the FLAGS to point to install location of the sfml library if we want to use SFML for image displaying
# CFLAGS = -std=c++17 -I/opt/homebrew/opt/sfml/include 	 
# LDFLAGS = -L/opt/homebrew/opt/sfml/lib -lsfml-graphics -lsfml-window -lsfml-system

## Uncomment these depending on the test you want to run
# SRCS = showPerlin.cpp perlin.cpp	## Displays the generated noise
SRCS = timePerlin.cpp perlin.cpp	## Times the generation of noise

OBJS = $(SRCS:.cpp=.o)
EXEC = perlin

$(EXEC): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

test: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(OBJS) $(EXEC)

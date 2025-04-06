CC = g++

## Update the FLAGS to point to install location of the sfml library
CFLAGS = -std=c++17 -I/opt/homebrew/opt/sfml/include 	 
LDFLAGS = -L/opt/homebrew/opt/sfml/lib -lsfml-graphics -lsfml-window -lsfml-system

SRCS = showPerlin.cpp perlin.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = showPerlin

$(EXEC): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

test: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(OBJS) $(EXEC)

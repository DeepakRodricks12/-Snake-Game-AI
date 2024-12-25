#Snake-Game-AI

The Snake game is a computer action game, whose goal is to control a snake to move and collect food on a map. The snake starts with a specific length. At each time step the snake moves one step, and can go straight, turn left, or turn right. Each time the snake eats the food, the length of the snake  increases. The game ends when the snake either hits its own body, the obstacles, or the boundary (3 hell states). The code effectively creates a snake game AI environment using Gymnasium and pygame, employing a simple greedy algorithm for the snake's movement. It simulates multiple games, tracks scores, and visualizes the results.

Key Components
Imports and Initialization: Various necessary modules and libraries are imported, including gymnasium, pygame, random, enum, namedtuple, and numpy. pygame is initialized, and a font is set for rendering text on the game screen.

Direction Enum and Point Named Tuple: Direction is an enumeration class representing the four possible directions: RIGHT, LEFT, UP, DOWN. Point is a named tuple representing the coordinates (x, y) on the game grid.

RGB Color Definitions: Various RGB color values are defined for different game elements, such as the snake, food,  obstacles, and background.

SnakeGameAI Class: Initialization (__init__): Sets up the game environment, including window dimensions, number of obstacles, action space, and observation space.

Reset (reset): Resets the game state, including the snake's initial position, direction, score, food, and obstacles.

Place Food and Obstacles (_place_food, _place_obstacles, _get_random_point): Methods to randomly place the food and obstacles on the grid, ensuring they do not overlap with the snake or each other.

Step (step): Executes a game step based on the provided action (or the best action if none is provided), updates the game state, calculates the reward, and checks for collisions or game over conditions.

Best Action (_get_best_action): Greedy algorithm to determine the best action based on the shortest distance to the food and avoiding collisions.

Utility Methods (_get_next_position, _get_distance, is_collision, _update_ui, _move, _get_state): Various helper methods to calculate distances, check for collisions, update the game display, move the snake, and get the current game state.

Simulation and Plotting: Simulate Game (simulate_game): Function to simulate multiple games (default is 500) using the greedy algorithm, collect scores, and print them.

Main Execution (if __name__ == '__main__'): Initializes the game environment, runs the simulation, plots the scores over the games, and closes the environment.

Execution Flow-Initialize the Game Environment: The SnakeGameAI class is instantiated with default parameters.

Run Simulations:The simulate_game function is called with the environment instance and the number of games to simulate.For each game, the environment is reset, and the game loop runs until the game is over. 
During each step, the best action is determined, the state is updated, rewards are accumulated, and the score is tracked.

import gymnasium as gym
from gym import spaces
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    """
    Enum class representing the direction of the snake.
    """
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 150, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

BLOCK_SIZE = 25
SPEED = 100

class SnakeGameAI(gym.Env):
    """
    Custom Gymnasium environment for the Snake game with AI.

    Attributes:
        w (int): Width of the game window.
        h (int): Height of the game window.
        num_obstacles (int): Number of obstacles in the game.
        action_space (gym.spaces.Discrete): Action space for the environment.
        observation_space (gym.spaces.Box): Observation space for the environment.
        display (pygame.Surface): Pygame display surface.
        clock (pygame.time.Clock): Pygame clock.
    """
    def __init__(self, w=800, h=800, num_obstacles=15):
        """
        Initializes the Snake game environment.

        Args:
            w (int): Width of the game window.
            h (int): Height of the game window.
            num_obstacles (int): Number of obstacles in the game.
        """
        super(SnakeGameAI, self).__init__()
        self.w = w
        self.h = h
        self.num_obstacles = num_obstacles

        # Define action space: [straight, right, left]
        self.action_space = spaces.Discrete(3)

        # Define observation space: 11 values (can be modified based on your observation design)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """
        Resets the game state.

        Returns:
            np.array: The initial state of the game.
        """
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.obstacles = []
        self._place_food()
        self._place_obstacles()
        self.frame_iteration = 0

        state = self._get_state()
        return state

    def _place_food(self):
        """
        Places the food randomly on the grid, ensuring it doesn't overlap with the snake or obstacles.
        """
        self.food = self._get_random_point(exclude=self.snake + self.obstacles)

    def _place_obstacles(self):
        """
        Places obstacles randomly on the grid, ensuring they don't overlap with the snake, food, or other obstacles.
        """
        self.obstacles = [self._get_random_point(exclude=self.snake + self.obstacles + [self.food]) for _ in range(self.num_obstacles)]

    def _get_random_point(self, exclude):
        """
        Generates a random point on the grid that does not overlap with the excluded points.

        Args:
            exclude (list): List of points to exclude.

        Returns:
            Point: A random point on the grid.
        """
        available_points = set(
            Point(x, y)
            for x in range(0, self.w, BLOCK_SIZE)
            for y in range(0, self.h, BLOCK_SIZE)
        ) - set(exclude)
        return random.choice(list(available_points))

    def step(self, action=None):
        """
        Executes a game step based on the given action.

        Args:
            action (int): The action to take (0: straight, 1: right, 2: left).

        Returns:
            tuple: (state, reward, done, info)
        """
        self.frame_iteration += 1

        # Use the greedy algorithm to determine the best action if no action is provided
        if action is None:
            action = self._get_best_action()

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -50
            return self._get_state(), reward, game_over, {}

        if self.head == self.food:
            self.score += 100  # Increment score by 100
            reward = 100       # Set reward to 100
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return self._get_state(), reward, game_over, {}

    def _get_best_action(self):
        """
        Determines the best action to take based on the greedy algorithm.

        Returns:
            int: The best action (0: straight, 1: right, 2: left).
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        # Current position of the head
        head = self.head

        # Positions for each possible action
        straight_pos = self._get_next_position(head, clock_wise[idx])
        right_pos = self._get_next_position(head, clock_wise[(idx + 1) % 4])
        left_pos = self._get_next_position(head, clock_wise[(idx - 1) % 4])

        # Calculate the distances to the food for each position
        dist_straight = self._get_distance(straight_pos, self.food)
        dist_right = self._get_distance(right_pos, self.food)
        dist_left = self._get_distance(left_pos, self.food)

        # Check for collisions for each position
        collision_straight = self.is_collision(straight_pos)
        collision_right = self.is_collision(right_pos)
        collision_left = self.is_collision(left_pos)

        # Choose the action with the smallest distance to the food that does not result in a collision
        if not collision_straight and (dist_straight <= dist_right or collision_right) and (dist_straight <= dist_left or collision_left):
            return 0  # Straight
        elif not collision_right and (dist_right < dist_straight or collision_straight) and (dist_right < dist_left or collision_left):
            return 1  # Right
        elif not collision_left:
            return 2  # Left
        else:
            # If all directions are risky, choose the safest option (avoid collisions)
            if not collision_straight:
                return 0
            elif not collision_right:
                return 1
            else:
                return 2

    def _get_next_position(self, current_pos, direction):
        """
        Gets the next position of the snake's head based on the current position and direction.

        Args:
            current_pos (Point): The current position of the snake's head.
            direction (Direction): The direction of movement.

        Returns:
            Point: The next position of the snake's head.
        """
        x = current_pos.x
        y = current_pos.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        return Point(x, y)

    def _get_distance(self, point1, point2):
        """
        Calculates the Manhattan distance between two points.

        Args:
            point1 (Point): The first point.
            point2 (Point): The second point.

        Returns:
            int: The Manhattan distance between the two points.
        """
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)

    def render(self, mode='human', close=False):
        """
        Renders the game state.

        Args:
            mode (str): The mode in which to render the game.
            close (bool): Whether to close the rendering window.
        """
        self._update_ui()
        self.clock.tick(SPEED)

    def close(self):
        """
        Closes the game environment.
        """
        pygame.quit()

    def is_collision(self, pt=None):
        """
        Checks if there is a collision at the given point.

        Args:
            pt (Point): The point to check for a collision.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        if pt in self.obstacles:
            return True
        return False

    def _update_ui(self):
        """
        Updates the game display.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()

        self.display.fill(GREEN2)

        for pt in self.snake:
            pygame.draw.rect(self.display, (255, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (255, 255, 150), pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        food_center = (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2)
        pygame.draw.polygon(self.display, BLUE, [
            (food_center[0], food_center[1] - BLOCK_SIZE // 2), 
            (food_center[0] + BLOCK_SIZE // 2, food_center[1]), 
            (food_center[0], food_center[1] + BLOCK_SIZE // 2), 
            (food_center[0] - BLOCK_SIZE // 2, food_center[1])
        ])

        for pt in self.obstacles:
            pygame.draw.rect(self.display, (200,0,0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Moves the snake based on the given action.

        Args:
            action (int): The action to take (0: straight, 1: right, 2: left).
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:
            new_dir = clock_wise[idx]
        elif action == 1:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _get_state(self):
        """
        Gets the current state of the game.

        Returns:
            np.array: The current state of the game.
        """
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y   # food down
        ]
        
        return np.array(state, dtype=np.float32)

def simulate_game(env, games=500):
    """
    Simulates multiple games and collects the scores.

    Args:
        env (SnakeGameAI): The game environment.
        games (int): Number of games to simulate.

    Returns:
        list: List of scores for each game.
    """
    scores = []
    for _ in range(games):
        state = env.reset()
        score = 0
        while True:
            action = env._get_best_action()  # Use the greedy action
            state, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
        print(f"Score: {score}")

    return scores

if __name__ == '__main__':
    env = SnakeGameAI()
    games = 5
    scores = simulate_game(env, games)

    # Plot the scores
    plt.plot(range(games), scores)
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.title('Scores over games')
    plt.show()

    env.close()

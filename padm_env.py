# Imports:
# --------
import gymnasium as gym
from gym import spaces
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    """Enumeration for snake direction."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 150, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

BLOCK_SIZE = 20
SPEED = 2000

class SnakeGameAI(gym.Env):
    """Snake game environment compatible with OpenAI Gym."""
    def __init__(self, w=800, h=800, num_obstacles=3):
        """
        Initialize the snake game environment.
        
        Parameters:
        w (int): Width of the game window.
        h (int): Height of the game window.
        num_obstacles (int): Number of obstacles in the game.
        """
        super(SnakeGameAI, self).__init__()
        self.w = w
        self.h = h
        self.num_obstacles = num_obstacles

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.float32)

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    # Method 1: .reset()
    # ---------
    def reset(self):
        """
        Reset the game to the initial state.
        
        Returns:
        np.array: Initial state of the game.
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.obstacles = []
        self._place_food()
        self._place_obstacles()
        self.frame_iteration = 0

        return self._get_state()

    def _place_food(self):
        """Place food at a random location on the game board."""
        self.food = self._get_random_point(exclude=self.snake + self.obstacles)

    def _place_obstacles(self):
        """Place obstacles at random locations on the game board."""
        self.obstacles = [self._get_random_point(exclude=self.snake + self.obstacles + [self.food]) for _ in range(self.num_obstacles)]

    def _get_random_point(self, exclude):
        """
        Get a random point on the game board that is not in the exclude list.
        
        Parameters:
        exclude (list): List of points to exclude.
        
        Returns:
        Point: A random point on the game board.
        """
        available_points = set(
            Point(x, y)
            for x in range(0, self.w, BLOCK_SIZE)
            for y in range(0, self.h, BLOCK_SIZE)
        ) - set(exclude)
        return random.choice(list(available_points))
    
    # Method 2: .step()
    # ---------
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Parameters:
        action (int): Action to be performed.
        
        Returns:
        tuple: Tuple containing the new state, reward, game over flag, and additional info.
        """
        self.frame_iteration += 1

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -50
            return self._get_state(), reward, game_over, {}

        if self.head == self.food:
            self.score += 1
            reward = 100
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return self._get_state(), reward, game_over, {}
    
    # Method 3: .render()
    # ---------
    def render(self, mode='human', close=False):
        """
        Render the game state to the display.
        
        Parameters:
        mode (str): Mode for rendering (default is 'human').
        close (bool): Flag to close the rendering (default is False).
        """
        self._update_ui()
        self.clock.tick(SPEED)

    def close(self):
        """Close the game and quit pygame."""
        pygame.quit()

    def is_collision(self, pt=None):
        """
        Check if the snake has collided with the boundaries, itself, or obstacles.
        
        Parameters:
        pt (Point): Point to check for collision (default is head of the snake).
        
        Returns:
        bool: True if collision occurred, False otherwise.
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
        """Update the game display."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.display.fill(GREEN2)

        for pt in self.snake:
            pygame.draw.rect(self.display, (255, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (255, 255, 150), pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        food_center = (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2)
        pygame.draw.circle(self.display, BLUE, food_center, BLOCK_SIZE // 2)

        for pt in self.obstacles:
            pygame.draw.rect(self.display, RED, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        Move the snake in the specified direction.
        
        Parameters:
        action (int): Action specifying the direction to move.
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 1:  # right turn -> down
            new_dir = clock_wise[(idx + 1) % 4]
        elif action == 2:  # left turn -> up
            new_dir = clock_wise[(idx - 1) % 4]
        else:  # no change
            new_dir = clock_wise[idx]

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
        Get the current state of the game. Returns: np.array: State of the game.
        """
        point_l = Point(self.head.x - BLOCK_SIZE, self.head.y)
        point_r = Point(self.head.x + BLOCK_SIZE, self.head.y)
        point_u = Point(self.head.x, self.head.y - BLOCK_SIZE)
        point_d = Point(self.head.x, self.head.y + BLOCK_SIZE)

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

            self.food.x < self.head.x,
            self.food.x > self.head.x,
            self.food.y < self.head.y,
            self.food.y > self.head.y
        ]

        return np.array(state, dtype=np.float32)
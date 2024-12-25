# Import:
# -------
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
    Enumeration for snake direction.
    """
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# RGB colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 150, 0)
BLACK = (0,0,0)
BROWN = (139, 69, 19)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

BLOCK_SIZE = 25
SPEED = 2000

class SnakeGameAI:
    """
    A class to represent the Snake game with AI capabilities.
    """

    def __init__(self, w=800, h=800, num_obstacles=3):
        """
        Initialize the game with given width, height, and number of obstacles.
        """
        self.w = w
        self.h = h
        self.num_obstacles = num_obstacles
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        """
        Reset the game to the initial state.
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


    def _place_food(self):
        """
        Place food at a random location not occupied by the snake.
        """
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _place_obstacles(self):
        """
        Place obstacles at random locations not occupied by the snake or food.
        """
        self.obstacles = []
        for _ in range(self.num_obstacles):
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            point = Point(x, y)
            while point in self.snake or point == self.food or point in self.obstacles:
                x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
                point = Point(x, y)
            self.obstacles.append(point)


    def play_step(self, action):
        """
        Play one step in the game based on the given action.

        Parameters:
            action (list): The action to be taken by the snake.

        Returns:
            reward (int): The reward for the action taken.
            game_over (bool): Whether the game is over or not.
            score (int): The current score.
        """
        self.frame_iteration += 1
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return game over and score
        return reward, game_over, self.score
    
    def render(self):
        """
        Render the game UI and update the display.
        """
        self._update_ui()
        self.clock.tick(SPEED)

    def is_collision(self, pt=None):
        """
        Check if there is a collision at the given point.

        Parameters:
            pt (Point): The point to check for collision.

        Returns:
            bool: Whether there is a collision or not.
        """
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        # Hits obstacles
        if pt in self.obstacles:
            return True

        return False


    def _update_ui(self):
        """
        Update the UI with the current state of the game.
        """
        self.display.fill(GREEN2)

        for pt in self.snake:
            pygame.draw.rect(self.display, (255, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, (255, 255, 150), pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw food as a blue diamond
        food_center = (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2)
        pygame.draw.polygon(self.display, BLUE, [
            (food_center[0], food_center[1] - BLOCK_SIZE // 2),  # Top
            (food_center[0] + BLOCK_SIZE // 2, food_center[1]),  # Right
            (food_center[0], food_center[1] + BLOCK_SIZE // 2),  # Bottom
            (food_center[0] - BLOCK_SIZE // 2, food_center[1])   # Left
        ])

        for pt in self.obstacles:
            pygame.draw.rect(self.display, (200,0,0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))  

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        """
        Move the snake in the direction based on the given action.

        Parameters:
            action (list): The action to be taken by the snake.
        """
        # [Straight, Right, Left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]              # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]         # Right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]         # Left turn r -> u -> l -> d

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



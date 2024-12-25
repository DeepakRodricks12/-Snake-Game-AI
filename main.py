# Import:
# -------
import torch
import random
import numpy as np
from collections import deque
from padm_env import SnakeGameAI, Direction, Point
from DQN_model import Linear_QNet, QTrainer
from utils import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    """
    The Agent class handles the decision-making process for the Snake game AI using a deep Q-network.

    Attributes:
        n_games (int): Number of games played.
        epsilon (float): Degree of randomness in actions for exploration.
        gamma (float): Discount rate for future rewards.
        memory (deque): Replay memory to store experiences.
        model (Linear_QNet): The neural network model.
        trainer (QTrainer): Trainer to handle the training of the model.
    """

    def __init__(self):
        """Initialize the Agent with its neural network and trainer."""
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when max memory is reached
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """
        Get the current state of the game.

        Args:
            game (SnakeGameAI): The current game instance.

        Returns:
            np.array: The state as a numpy array of binary values.
        """
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # Food Left
            game.food.x > game.head.x,  # Food Right
            game.food.y < game.head.y,  # Food Up
            game.food.y > game.head.y   # Food Down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience in the replay memory.

        Args:
            state (np.array): The current state.
            action (list): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state.
            done (bool): Whether the game is done.
        """
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        """
        Train the neural network with a batch of experiences from the replay memory.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the neural network with a single experience tuple.

        Args:
            state (np.array): The current state.
            action (list): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state.
            done (bool): Whether the game is done.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Determine the action to take based on the current state.

        Args:
            state (np.array): The current state.

        Returns:
            list: The action to take as a one-hot encoded list.
        """
        # Random moves: Tradeoff - Exploration / Exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """
    Train the Snake game AI by playing multiple games and improving the model based on the experiences.
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            

if __name__ == '__main__':
    train()

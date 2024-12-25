# Imports:
# --------
from padm_env import SnakeGameAI
from q_learning import train_q_learning, visualize_q_table
import numpy as np

# User definitions:
train = True
visualize_results = True

# Optimized Hyperparameters:
learning_rate = 0.1            # Learning rate
gamma = 0.95                   # Discount factor
epsilon = 1.0                  # Exploration rate
epsilon_min = 0.01             # Minimum exploration rate
epsilon_decay = 0.995          # Decay rate for exploration
no_episodes = 500              # Number of episodes

# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = SnakeGameAI()
    q_table = train_q_learning(env, 
                               episodes=no_episodes, 
                               alpha=learning_rate, 
                               gamma=gamma, 
                               epsilon_start=epsilon, 
                               epsilon_decay=epsilon_decay, 
                               min_epsilon=epsilon_min)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(q_values_path="q_table.npy")

    

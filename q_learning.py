# Imports:
# --------
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env, 
                     episodes, 
                     alpha, 
                     gamma, 
                     epsilon_start, 
                     epsilon_decay, 
                     min_epsilon):
    
    # Initialize the Q-table:
    # -----------------------
    q_table = {}
    epsilon = epsilon_start
    
    # Q-learning algorithm:
    # ---------------------

    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(episodes):
        state = tuple(env.reset())  
        state = tuple(state)
        total_reward = 0
        done = False
        
        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while not done:

            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                q_values = q_table.get(state, np.zeros(env.action_space.n))
                action = np.argmax(q_values)        # Exploit

            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = tuple(next_state)  
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)
            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)

            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state

            #! Step 5: Stop the episode if the agent reaches Hell-states
            #! -------
            if done:
                break
        
        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save('q_table.npy', q_table)
    print("Saved the Q-table.")

    return q_table

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(actions = ['Right', 'Down', 'Left', 'Up'],
                      q_values_path="q_table.npy"):
    
    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path, allow_pickle=True).item()

        # Create subplots for each action:
        # --------------------------------
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle('Q-table for each Action', fontsize=16)
    
        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = np.zeros((4, 4))

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)
        
            for state, values in q_table.items():
                if isinstance(state, tuple) and len(state) >= 2:
                    x, y = int(state[0]), int(state[1])
                    if 0 <= x < 4 and 0 <= y < 4:
                        heatmap_data[x, y] = values[i]

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", 
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})
        
            ax.set_title(f'Action: {actions[i]}')
    
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

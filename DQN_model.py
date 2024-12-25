# Import:
# -------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Deep Q-Network:
# ---------------
class Linear_QNet(nn.Module):
    """
    A simple linear neural network model for Q-learning.
    
    Attributes:
        linear1 (nn.Linear): The first linear layer.
        linear2 (nn.Linear): The second linear layer.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the linear neural network with one hidden layer.
        
        Args:
            input_size (int): The size of the input layer.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output layer.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Saves the model state to a file.
        
        Args:
            file_name (str, optional): The name of the file to save the model. Defaults to 'model.pth'.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    A trainer class for training a Q-learning model.
    
    Attributes:
        lr (float): The learning rate.
        gamma (float): The discount factor.
        model (nn.Module): The neural network model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (nn.Module): The loss function.
    """
    def __init__(self, model, lr, gamma):
        """
        Initializes the QTrainer with a model, learning rate, and discount factor.
        
        Args:
            model (nn.Module): The neural network model to be trained.
            lr (float): The learning rate.
            gamma (float): The discount factor.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Performs a single training step for the model.
        
        Args:
            state (array-like): The current state.
            action (array-like): The action taken.
            reward (float): The reward received.
            next_state (array-like): The next state.
            done (bool): Whether the episode is done.
        """
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

import random
from collections import deque

# this file contains the model and the trainer class
# this model is used to train the agent
# the input of this model is a tuple of list with 2 number in each list representing the moves in 1 to 8 example [(1, 2), (3, 8)]
# the output layer has to output a tuple of 2 numbers representing the move to play
# the model is a simple neural network with 2 hidden layers with 256 neurons each
# the model is trained using the Q-learning algorithm


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x.view(x.size(0), -1))) # flatten input tensor
        x = self.linear2(x)
        return x
    

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, state, action, reward, next_state, done):
        # Convert state and next_state to tensors
        print("state: ", state)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        # Calculate the predicted Q-values for the current state and next state
        current_q_values = self.model(state_tensor)
        next_q_values = self.target_model(next_state_tensor)

        # Select the Q-value for the action taken in the current state
        action_indices = torch.tensor(action, dtype=torch.int64)
        current_q_values_for_action = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        # Calculate the target Q-value for the current state using the Bellman equation
        target_q_values = reward + self.discount_factor * next_q_values.max(1)[0] * (1 - done)

        # Calculate the loss between the predicted and target Q-values
        loss = self.loss_function(current_q_values_for_action, target_q_values.detach())

        # Update the weights of the model using backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_target_model()
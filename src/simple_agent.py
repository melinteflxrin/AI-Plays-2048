"""
DQN Agent for 2048 game
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQNAgent:
    """
    Deep Q-Network agent that learns to play 2048
    """
    
    def __init__(self, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # check if GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # exploration rate -> how often to try random moves
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # creating the neural network
        self.q_network = self._build_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # memory
        self.memory = deque(maxlen=10000)  # remember last 10000 moves
        
    def _build_network(self):
        """
        neural network that:
        - takes 4x4 board as input (16 numbers)
        - outputs 4 action values (up, down, left, right)
        """
        network = nn.Sequential(
            nn.Flatten(),
            
            # hidden layers
            nn.Linear(16, 128),   # 16 inputs → 128 neurons
            nn.ReLU(),            # Activation function
            nn.Linear(128, 128),  # 128 → 128 neurons  
            nn.ReLU(),
            nn.Linear(128, 64),   # 128 → 64 neurons
            nn.ReLU(),
            nn.Linear(64, 4)      # 64 → 4 outputs (one for each action)
        )
        
        return network.to(self.device)
    
    def choose_action(self, observation):
        """
        choose action based on the current board state
        uses epsilon-greedy: sometimes random (explore), sometimes best known (exploit)
        """
        # random action for exploration
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # random action (0=up, 1=down, 2=left, 3=right)
        
        # use neural network to choose best action
        with torch.no_grad():
            # convert observation to tensor
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # get Q-values for all actions
            q_values = self.q_network(state_tensor)

            # choose action with highest Q-value
            action = q_values.argmax().item()
            
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        store experience in memory for later learning
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, batch_size=32):
        """
        learn from past experiences stored in memory
        """
        # need enough experiences to learn from
        if len(self.memory) < batch_size:
            return

        # sample random batch of experiences
        batch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        # current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (0.95 * next_q_values * ~dones)  # 0.95 = discount factor

        # calculate loss -> how wrong were we?
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon (explore less over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """save the trained model"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),     # <- knowledge is stored here
            'optimizer_state_dict': self.optimizer.state_dict(), # <- saves learning state
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")
        
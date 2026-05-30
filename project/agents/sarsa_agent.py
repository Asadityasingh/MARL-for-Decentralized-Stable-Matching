"""
SARSA Agent with Neural Network Function Approximation
Implements on-policy SARSA as described in the paper
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Optional, Tuple


class QNetwork(nn.Module):
    """
    Multi-layer perceptron for Q-value approximation.
    Architecture: input -> 50 -> 25 -> output
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class SARSAAgent:
    """
    SARSA agent with experience replay and epsilon-greedy exploration.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, agent_id: str,
                 lr: float = 1e-4, gamma: float = 0.9, 
                 epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995, replay_buffer_size: int = 10,
                 device: str = 'cpu'):
        """
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            agent_id: Unique identifier for the agent
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay factor (non-linear)
            replay_buffer_size: Number of recent episodes to keep
            device: 'mps', 'cuda', or 'cpu'
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        # Q-network
        self.q_network = QNetwork(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer (stores recent episodes)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.current_episode = []
        
        # For SARSA (need to track current state-action)
        self.last_state = None
        self.last_action = None
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            training: If True, use epsilon-greedy; else greedy
        
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, 
                        next_action: int, done: bool):
        """
        Store SARSA transition (s, a, r, s', a') in current episode.
        """
        self.current_episode.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_action': next_action,
            'done': done
        })
    
    def end_episode(self):
        """
        End current episode and add to replay buffer.
        """
        if len(self.current_episode) > 0:
            self.replay_buffer.append(self.current_episode)
            self.current_episode = []
    
    def update(self, batch_size: int = 32):
        """
        Update Q-network using SARSA with experience replay.
        Samples from recent episodes only.
        """
        if len(self.replay_buffer) == 0:
            return 0.0
        
        # Sample transitions from recent episodes
        all_transitions = []
        for episode in self.replay_buffer:
            all_transitions.extend(episode)
        
        if len(all_transitions) < batch_size:
            batch_size = len(all_transitions)
        
        # Random sample
        indices = np.random.choice(len(all_transitions), batch_size, replace=False)
        batch = [all_transitions[i] for i in indices]
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([t['state'] for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t['action'] for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t['reward'] for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t['next_state'] for t in batch])).to(self.device)
        next_actions = torch.LongTensor(np.array([t['next_action'] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t['done'] for t in batch], dtype=np.float32)).to(self.device)
        
        # SARSA update: Q(s,a) <- Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values (using next_action, not max)
        with torch.no_grad():
            next_q = self.q_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon (non-linear decay)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

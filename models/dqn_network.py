"""
Deep Q-Network for KG error detection and correction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.2):
        """
        Initialize DQN
        
        Args:
            input_dim: Input dimension (state size)
            output_dim: Output dimension (action size)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super(DQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for all actions
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: State tensor
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if torch.rand(1).item() < epsilon:
            # Random action
            return torch.randint(0, self.output_dim, (1,)).item()
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()


class DuelingDQN(nn.Module):
    """Dueling DQN architecture"""
    
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 dropout_rate: float = 0.2):
        """
        Initialize Dueling DQN
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (action size)
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super(DuelingDQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for all actions
        """
        features = self.shared(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy
        
        Args:
            state: State tensor
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if torch.rand(1).item() < epsilon:
            # Random action
            return torch.randint(0, self.output_dim, (1,)).item()
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int):
        """
        Sample batch from buffer
        
        Args:
            batch_size: Batch size
            
        Returns:
            Batch of experiences
        """
        import random
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([s for s, _, _, _, _ in batch])
        actions = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long)
        rewards = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32)
        next_states = torch.stack([ns for _, _, _, ns, _ in batch])
        dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
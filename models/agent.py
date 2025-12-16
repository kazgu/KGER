"""
DQN Agent for KG error detection and correction
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple

from models.dqn_network import DQN, DuelingDQN, ReplayBuffer
from models.simple_embedding_adapter import SimpleEmbeddingAdapter, create_embedding_adapter
from config import Config


class DQNAgent:
    """DQN Agent for KG error correction"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 kg_model=None, use_dueling: bool = False,
                 device: str = 'cpu'):
        """
        Initialize DQN Agent
        
        Args:
            state_dim: State dimension 
            action_dim: Action dimension
            kg_model: KG model for embedding processing (needed for RotatE)
            use_dueling: Whether to use dueling DQN
            device: Device to use
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.kg_model = kg_model
        
        # 简单的embedding adapter（如果启用）
        self.adapter = None
        if Config.EmbeddingAdapter.use_adapter and kg_model is not None:
            self.adapter = create_embedding_adapter(
                kg_model=kg_model,
                output_dim=Config.EmbeddingAdapter.output_dim
            ).to(self.device)
            print(f"   Created simple embedding adapter: auto -> {Config.EmbeddingAdapter.output_dim}D")
        else:
            print(f"   No adapter - using direct embeddings")
        
        # Create networks
        if use_dueling:
            self.q_network = DuelingDQN(
                state_dim, action_dim,
                hidden_dims=Config.DQN.hidden_dims,
                dropout_rate=Config.DQN.dropout_rate
            ).to(self.device)
            
            self.target_network = DuelingDQN(
                state_dim, action_dim,
                hidden_dims=Config.DQN.hidden_dims,
                dropout_rate=Config.DQN.dropout_rate
            ).to(self.device)
        else:
            self.q_network = DQN(
                state_dim, action_dim,
                hidden_dims=Config.DQN.hidden_dims,
                dropout_rate=Config.DQN.dropout_rate
            ).to(self.device)
            
            self.target_network = DQN(
                state_dim, action_dim,
                hidden_dims=Config.DQN.hidden_dims,
                dropout_rate=Config.DQN.dropout_rate
            ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer - DQN + adapter parameters (if enabled)
        optimizer_params = list(self.q_network.parameters())
        if self.adapter is not None:
            optimizer_params.extend(list(self.adapter.parameters()))
            print(f"   Optimizer includes adapter parameters")
        
        self.optimizer = optim.Adam(
            optimizer_params,
            lr=Config.DQN.learning_rate
        )
        
        # Replay buffer
        self.memory = ReplayBuffer(Config.DQN.memory_size)
        
        # Training parameters
        self.gamma = Config.DQN.gamma
        self.epsilon = Config.DQN.epsilon_start
        self.epsilon_min = Config.DQN.epsilon_end
        self.epsilon_decay = Config.DQN.epsilon_decay
        
        # Counters
        self.steps_done = 0
        self.episodes_done = 0
        
        # Training history
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
    def get_state_representation(self, head_idx: int, relation_idx: int, tail_idx: int,
                                additional_features: Optional[torch.Tensor] = None, training: bool = True) -> torch.Tensor:
        """
        Get state representation using direct embeddings

        Args:
            head_idx: Head entity index
            relation_idx: Relation index
            tail_idx: Tail entity index
            additional_features: Additional features (score, confidence, etc.)
            training: Whether in training mode

        Returns:
            State tensor for DQN
        """
        if self.kg_model is not None:
            # 根据training参数控制梯度计算
            if training:
                h_emb, r_emb, t_emb = self.kg_model.get_embeddings(head_idx, relation_idx, tail_idx)
            else:
                with torch.no_grad():
                    h_emb, r_emb, t_emb = self.kg_model.get_embeddings(head_idx, relation_idx, tail_idx)
            
            # Prepare additional features
            if additional_features is None:
                score = self.kg_model.score_triple(head_idx, relation_idx, tail_idx)
                additional_features = torch.tensor([
                    score,  # Triple score
                    0.0, 0.0, 0.0,  # Placeholder confidence features
                    0.0, 0.0, 0.0,  # Placeholder type features  
                    0.0, 0.0, 0.0   # Placeholder other features
                ], device=self.device, dtype=torch.float32)
            
            # 统一处理embeddings - 通过adapter或截断
            if self.adapter is not None:
                # 使用adapter进行维度转换（最直接最简单的方式）
                h_emb = self.adapter(h_emb.flatten())
                r_emb = self.adapter(r_emb.flatten())
                t_emb = self.adapter(t_emb.flatten())
            else:
                # fallback: 截断到标准维度
                max_emb_dim = 128
                h_emb = h_emb.flatten()[:max_emb_dim]
                r_emb = r_emb.flatten()[:max_emb_dim] 
                t_emb = t_emb.flatten()[:max_emb_dim]
                
                # Pad if necessary
                if h_emb.shape[0] < max_emb_dim:
                    h_emb = torch.cat([h_emb, torch.zeros(max_emb_dim - h_emb.shape[0], device=self.device)])
                if r_emb.shape[0] < max_emb_dim:
                    r_emb = torch.cat([r_emb, torch.zeros(max_emb_dim - r_emb.shape[0], device=self.device)])
                if t_emb.shape[0] < max_emb_dim:
                    t_emb = torch.cat([t_emb, torch.zeros(max_emb_dim - t_emb.shape[0], device=self.device)])
            
            # 拼接所有组件
            state = torch.cat([
                h_emb,
                r_emb, 
                t_emb,
                additional_features
            ])
            
            return state.to(self.device)
        else:
            raise ValueError("No KG model provided for state representation")
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Random action
            action = np.random.randint(0, self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(self, state: torch.Tensor, action: int, 
                        reward: float, next_state: torch.Tensor, done: bool):
        """
        Store transition in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        # Detach states from gradient graph before storing to avoid double backward
        # This is crucial when using adapter since states retain gradient info
        state_detached = state.detach() if isinstance(state, torch.Tensor) else state
        next_state_detached = next_state.detach() if isinstance(next_state, torch.Tensor) else next_state
        
        self.memory.push(state_detached, action, reward, next_state_detached, done)
        
    def train_step(self, batch_size: int = 32) -> Optional[Tuple[float, float]]:
        """
        Perform one training step

        Args:
            batch_size: Batch size

        Returns:
            DQN loss
        """
        if len(self.memory) < batch_size:
            return None

        # Set adapter to training mode if present
        if self.adapter is not None:
            self.adapter.train()

        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute DQN loss
        dqn_loss = nn.MSELoss()(current_q_values, target_q_values)


        total_loss = dqn_loss


        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for DQN + adapter parameters
        all_params = list(self.q_network.parameters())
        if self.adapter is not None:
            all_params.extend(list(self.adapter.parameters()))
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        self.optimizer.step()

        # Record DQN loss
        dqn_loss_value = dqn_loss.item()
        self.loss_history.append(dqn_loss_value)

        # Return only DQN loss (no adapter loss)
        return dqn_loss_value
    
    def update_target_network(self):
        """Update target network with current Q network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        
    def save(self, path: str):
        """
        Save agent state
        
        Args:
            path: Save path
        """
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'epsilon_history': self.epsilon_history
        }, path)
        
    def load(self, path: str):
        """
        Load agent state
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.loss_history = checkpoint.get('loss_history', [])
        self.reward_history = checkpoint.get('reward_history', [])
        self.epsilon_history = checkpoint.get('epsilon_history', [])
        
    def get_q_values(self, state: torch.Tensor) -> np.ndarray:
        """
        Get Q-values for a state
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.cpu().numpy().squeeze()
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """
        Train for one episode
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward and number of steps
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step(Config.DQN.batch_size)
            
            # Update counters
            total_reward += reward
            steps += 1
            self.steps_done += 1
            
            # Update target network
            if self.steps_done % Config.DQN.target_update_freq == 0:
                self.update_target_network()
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        self.decay_epsilon()
        self.episodes_done += 1
        self.reward_history.append(total_reward)
        
        return total_reward, steps
    
    def evaluate(self, env, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate agent performance
        
        Args:
            env: Environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            Average reward and average improvement
        """
        total_rewards = []
        total_improvements = []
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            
            while True:
                # Select action (no exploration)
                action = self.select_action(state, training=False)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                
                if 'improvement' in info:
                    total_improvements.append(info['improvement'])
                
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
        
        avg_reward = np.mean(total_rewards)
        avg_improvement = np.mean(total_improvements) if total_improvements else 0.0
        
        return avg_reward, avg_improvement
    

    
    
    def save_model(self, path: str):
        """Save model + adapter (if present)"""
        save_dict = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'epsilon': self.epsilon
        }
        
        # Include adapter if present
        if self.adapter is not None:
            save_dict['adapter'] = self.adapter.state_dict()
            
        torch.save(save_dict, path)
    
    def load_model(self, path: str):
        """Load model + adapter (if present)"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        
        # Load adapter if present in checkpoint and agent has adapter
        if 'adapter' in checkpoint and self.adapter is not None:
            self.adapter.load_state_dict(checkpoint['adapter'])
            print(f"   Adapter loaded successfully")
        elif 'adapter' in checkpoint:
            print(f"   Warning: Checkpoint contains adapter but agent has no adapter")
        elif self.adapter is not None:
            print(f"   Warning: Agent has adapter but checkpoint has no adapter data")
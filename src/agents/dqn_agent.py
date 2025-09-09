"""
Deep Q-Network (DQN) Agent for Task Scheduling Optimization
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DuelingDQN(nn.Module):
    """Dueling DQN architecture for better value estimation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature extraction with deeper network
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Dueling DQN architecture
        self.value_stream = nn.Linear(hidden_dim // 2, 1)
        self.advantage_stream = nn.Linear(hidden_dim // 2, action_dim)
        
    def forward(self, x):
        # Check for NaN/Inf values and replace them
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        features = self.feature(x)
        
        # Dueling DQN
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for resource allocation optimization"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 5e-4, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.998, buffer_size: int = 50000,
                 batch_size: int = 64, target_update_freq: int = 5,
                 exploration_strategy: str = 'epsilon_greedy'):
        """
        Initialize DQN Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_end: Minimum exploration rate
            batch_size: Batch size for training
            buffer_size: Replay buffer capacity
            target_update_freq: Frequency of target network updates
            exploration_strategy: Exploration strategy (epsilon_greedy, boltzmann, noisy_nets)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Exploration strategy settings
        self.exploration_strategy = exploration_strategy
        self.use_boltzmann = (exploration_strategy == 'boltzmann')
        self.noisy_nets = (exploration_strategy == 'noisy_nets')
        self.action_counts = [0] * action_dim  # Track action usage
        
        # Networks
        self.q_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.max_grad_norm = 1.0  # For gradient clipping
        
        # Prioritized Experience Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.priority_alpha = 0.6  # Priority exponent
        self.priority_beta = 0.4   # Importance sampling weight
        self.priority_beta_increment = 0.001
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using advanced exploration strategies"""
        if training:
            # Boltzmann exploration (softmax with temperature)
            if hasattr(self, 'use_boltzmann') and self.use_boltzmann:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    
                    # Apply temperature-based softmax
                    temperature = max(0.5, self.epsilon * 2)  # Dynamic temperature
                    probabilities = torch.softmax(q_values / temperature, dim=1)
                    action = torch.multinomial(probabilities, 1).item()
                    return action
            
            # Epsilon-greedy with decay
            elif random.random() < self.epsilon:
                # Prioritized random exploration based on action history
                if hasattr(self, 'action_counts'):
                    # Choose less-explored actions more often
                    exploration_probs = 1.0 / (np.array(self.action_counts) + 1)
                    exploration_probs /= exploration_probs.sum()
                    return np.random.choice(self.action_dim, p=exploration_probs)
                else:
                    return random.randint(0, self.action_dim - 1)
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Add noise for exploration even during exploitation
            if training and hasattr(self, 'noisy_nets') and self.noisy_nets:
                noise = torch.randn_like(q_values) * 0.1 * self.epsilon
                q_values += noise
            
            return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer with priority"""
        # Calculate TD error for prioritization
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            current_q = self.q_network(state_tensor)[0, action].item()
            next_q = self.target_network(next_state_tensor).max(1)[0].item()
            target = reward + self.gamma * next_q * (1 - done)
            td_error = abs(target - current_q)
        
        # Store with priority
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.priorities.append(float(td_error + 1e-6))  # Ensure it's a float
        
        # Update action count for exploration
        if hasattr(self, 'action_counts'):
            self.action_counts[action] += 1
    
    def train_step(self):
        """Perform one training step with prioritized sampling"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Prioritized sampling
        if len(self.priorities) > 0:
            # Ensure all priorities are floats
            priorities = np.array([float(p) for p in self.priorities], dtype=np.float32)
            probabilities = priorities ** self.priority_alpha
            probabilities /= probabilities.sum()
            
            indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probabilities)
            batch = [self.replay_buffer[i] for i in indices]
            
            # Importance sampling weights
            weights = (len(self.replay_buffer) * probabilities[indices]) ** (-self.priority_beta)
            weights /= weights.max()
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.replay_buffer, self.batch_size)
            indices = None
            weights = torch.ones(self.batch_size).to(self.device)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute weighted loss
        td_errors = F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        loss = (weights * td_errors).mean()
        
        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update priorities for sampled experiences
        if indices is not None and len(self.priorities) > 0:
            td_errors_np = td_errors.cpu().detach().numpy()
            for i, idx in enumerate(indices):
                self.priorities[idx] = float(abs(td_errors_np[i]) + 1e-6)
        
        # Update training step counter
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Store loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_epsilon(self):
        """Update exploration rate and other parameters"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes += 1
        
        # Update importance sampling beta
        if hasattr(self, 'priority_beta'):
            self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        
        # Decay action counts for continued exploration
        if hasattr(self, 'action_counts') and self.episodes % 10 == 0:
            self.action_counts = [max(0, count * 0.95) for count in self.action_counts]
    
    def save_model(self, filepath: str):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes': self.episodes,
            'steps': self.steps
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.training_step = checkpoint.get('training_step', 0)
    
    def get_metrics(self) -> dict:
        """Get training metrics"""
        return {
            'episode_count': self.episode_count,
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0
        }

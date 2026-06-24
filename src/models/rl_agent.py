# src/models/rl_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

class TradingEnvironment(gym.Env):
    """Custom trading environment for RL agent"""
    
    def __init__(self, config: Dict, data: np.ndarray):
        super().__init__()
        self.config = config
        self.data = data
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity = 10000  # Starting equity
        
        # Action space: 0=hold, 1=long, 2=short
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(data.shape[1],),
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.seq_len
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity = 10000
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        self.current_step += 1
        current_price = self.data[self.current_step, 0]  # Close price
        
        # Execute action
        reward = 0
        done = False
        
        if action == 1:  # Long
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:  # Close short
                reward = (self.entry_price - current_price) / current_price
                self.trades.append({
                    'type': 'short',
                    'entry': self.entry_price,
                    'exit': current_price,
                    'pnl': reward
                })
                self.equity *= (1 + reward)
                self.position = 1
                self.entry_price = current_price
                
        elif action == 2:  # Short
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:  # Close long
                reward = (current_price - self.entry_price) / current_price
                self.trades.append({
                    'type': 'long',
                    'entry': self.entry_price,
                    'exit': current_price,
                    'pnl': reward
                })
                self.equity *= (1 + reward)
                self.position = -1
                self.entry_price = current_price
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            done = True
            # Close any open position
            if self.position != 0:
                final_price = self.data[-1, 0]
                if self.position == 1:
                    reward = (final_price - self.entry_price) / self.entry_price
                else:
                    reward = (self.entry_price - final_price) / self.entry_price
                self.equity *= (1 + reward)
        
        obs = self._get_observation()
        info = {
            'equity': self.equity,
            'position': self.position,
            'trades': len(self.trades)
        }
        
        return obs, reward, done, False, info
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step < self.seq_len:
            # Pad with zeros at beginning
            obs = np.zeros((self.seq_len, self.data.shape[1]))
            obs[-self.current_step-1:] = self.data[:self.current_step+1]
        else:
            obs = self.data[self.current_step - self.seq_len:self.current_step + 1]
        
        # Flatten the observation
        obs = obs.flatten().astype(np.float32)
        return obs

class RLAgentManager:
    def __init__(self, config: Dict):
        self.config = config
        self.agent_type = config.get('rl_agent', 'ppo')
        self.model = None
        self.env = None
        
    def train(self, data: np.ndarray, total_timesteps: int = 100000):
        """Train the RL agent"""
        # Create environment
        self.env = TradingEnvironment(self.config, data)
        check_env(self.env)
        
        # Initialize agent
        if self.agent_type == 'ppo':
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
        elif self.agent_type == 'sac':
            self.model = SAC(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=10000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                ent_coef='auto'
            )
        
        # Train
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(f"models/rl_{self.agent_type}.pkl")
        
        return self.model
    
    def predict(self, observation: np.ndarray) -> Tuple[int, Dict]:
        """Make prediction with trained agent"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
    def load_model(self, path: str):
        """Load trained model"""
        if self.agent_type == 'ppo':
            self.model = PPO.load(path)
        elif self.agent_type == 'sac':
            self.model = SAC.load(path)
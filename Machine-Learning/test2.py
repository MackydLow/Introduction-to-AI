# test.py

import torch
import numpy as np
from policy_model2 import PolicyNetwork
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
import gymnasium as gym
from torch.distributions import Categorical

# -------------------------------
# Simple Custom Environment
# -------------------------------
class SimpleGridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 10
        self.start = 0
        self.goal = self.grid_size - 1
        self.max_steps = 20
        self.current_step = 0

        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1,
                                            shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.position = self.start
        self.current_step = 0
        return np.array([self.position], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1

        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 1:
            self.position = min(self.grid_size - 1, self.position + 1)

        if self.position == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False

        if self.current_step >= self.max_steps:
            done = True

        return np.array([self.position], dtype=np.float32), reward, done, False, {}

# -------------------------------
# RL Helper Functions
# -------------------------------
def calculate_stepwise_returns(rewards, discount_factor=0.99):
    returns = []
    total_rewards = 0
    for reward in reversed(rewards):
        total_rewards = reward + discount_factor * total_rewards
        returns.insert(0, total_rewards)

    returns = torch.tensor(returns, dtype=torch.float32)
    return (returns - returns.mean()) / (returns.std() + 1e-8)

def forward_pass(env, policy, discount_factor=0.99):
    log_probs = []
    rewards = []
    done = False
    episode_return = 0

    observation, _ = env.reset()
    policy.train()

    while not done:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        probs = policy(obs_tensor)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        observation, reward, done, truncated, _ = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(reward)
        episode_return += reward

    log_probs = torch.cat(log_probs)
    returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, returns, log_probs

def calculate_loss(returns, log_probs):
    return -(returns * log_probs).sum()

def update_policy(returns, log_probs, optimizer):
    returns = returns.detach()
    loss = calculate_loss(returns, log_probs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# -------------------------------
# Training Loop
# -------------------------------
def main():

    print("Main started.")
    
    env = SimpleGridEnv()
    print("Environment created.")
    
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    print("Input/Output dimensions:", input_dim, output_dim)
    env = SimpleGridEnv()

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = PolicyNetwork(input_dim, hidden_dim=16, output_dim=output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    max_episodes = 500
    reward_threshold = 8
    n_trials = 20
    episode_returns = []

    for episode in range(1, max_episodes + 1):
        ep_ret, returns, log_probs = forward_pass(env, policy)
        update_policy(returns, log_probs, optimizer)

        episode_returns.append(ep_ret)
        mean_return = np.mean(episode_returns[-n_trials:])

        if episode % 20 == 0:
            print(f"Episode {episode}, Mean Reward = {mean_return:.2f}")

        if mean_return >= reward_threshold:
            print(f"Solved in {episode} episodes!")
            break

    torch.save(policy.state_dict(), "grid_policy.pt")
    print("Saved policy to grid_policy.pt")

if __name__ == "__main__":
    main()

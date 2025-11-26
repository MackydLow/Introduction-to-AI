import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
import gymnasium as gym

# ----------------------------
# 1. Define 2D Grid Environment
# ----------------------------
class GridWorld(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size=5, render_mode=None):
        super().__init__()
        self.size = size
        self.observation_space = gym.spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        self.max_steps = size * size
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        self.agent_pos = np.array([0, 0])
        self.steps_taken = 0
        return self.agent_pos.astype(np.float32), {}

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.size - 1, y + 1)

        self.agent_pos = np.array([x, y])
        self.steps_taken += 1

        done = np.array_equal(self.agent_pos, self.goal_pos) or self.steps_taken >= self.max_steps
        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else -0.01

        return self.agent_pos.astype(np.float32), reward, done, False, {}

    def render(self):
        grid = np.full((self.size, self.size), " . ")
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[ax, ay] = " A "
        grid[gx, gy] = " G "
        print("\n".join("".join(row) for row in grid))
        print()

# ----------------------------
# 2. Define Policy Network
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# 3. Helper Functions
# ----------------------------
def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    total = 0
    for r in reversed(rewards):
        total = r + discount_factor * total
        returns.insert(0, total)
    returns = torch.tensor(returns, dtype=torch.float32)
    normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return normalized_returns

def forward_pass(env, policy, discount_factor):
    log_probs = []
    rewards = []
    done = False
    episode_return = 0

    observation, _ = env.reset()
    while not done:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        action_probs = policy(obs_tensor)
        dist_cat = dist.Categorical(action_probs)
        action = dist_cat.sample()
        log_prob = dist_cat.log_prob(action)

        observation, reward, done, _, _ = env.step(action.item())
        log_probs.append(log_prob)
        rewards.append(reward)
        episode_return += reward

    log_probs = torch.cat(log_probs)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)
    return episode_return, stepwise_returns, log_probs

def calculate_loss(stepwise_returns, log_probs):
    return -(stepwise_returns * log_probs).sum()

def update_policy(stepwise_returns, log_probs, optimizer):
    loss = calculate_loss(stepwise_returns.detach(), log_probs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ----------------------------
# 4. Training Loop
# ----------------------------
def main():
    # Hyperparameters
    HIDDEN_DIM = 128
    DROPOUT = 0.2
    MAX_EPOCHS = 1000
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 20
    REWARD_THRESHOLD = 0.9
    LEARNING_RATE = 0.01
    PRINT_INTERVAL = 20

    env = GridWorld(size=5)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy = PolicyNetwork(input_dim, HIDDEN_DIM, output_dim, DROPOUT)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    episode_returns = []

    for episode in range(1, MAX_EPOCHS+1):
        ep_return, stepwise_returns, log_probs = forward_pass(env, policy, DISCOUNT_FACTOR)
        loss = update_policy(stepwise_returns, log_probs, optimizer)

        episode_returns.append(ep_return)
        mean_return = np.mean(episode_returns[-N_TRIALS:])

        if episode % PRINT_INTERVAL == 0:
            print(f"Episode {episode:3d} | Loss: {loss:.3f} | Mean Reward: {mean_return:.3f}")

        if mean_return >= REWARD_THRESHOLD:
            print(f"Solved in {episode} episodes! Mean Reward: {mean_return:.3f}")
            break

    torch.save(policy.state_dict(), "grid_policy.pt")
    print("Policy saved to grid_policy.pt")

    # ----------------------------
    # Visualization (fixed)
    # ----------------------------
    observation, _ = env.reset()
    done = False
    print("Visualizing agent path:")
    env.render()

    while not done:
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        action_probs = policy(obs_tensor)
        # Sample from policy so the agent actually moves
        dist_cat = torch.distributions.Categorical(action_probs)
        action = dist_cat.sample().item()
        observation, _, done, _, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    main()

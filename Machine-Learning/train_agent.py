# Adapted from https://www.datacamp.com/tutorial/policy-gradient-theorem to use with LunarLander problem

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as nnf
import torch.distributions as dist

import numpy as np

import gymnasium as gym  

import policy_model as pm


HIDDEN_DIM = 128
DROPOUT = 0.2

MAX_EPOCHS = 1500 
DISCOUNT_FACTOR = 0.99
N_TRIALS = 20
REWARD_THRESHOLD = 250 
PRINT_INTERVAL = 10

LEARNING_RATE = 0.003
LEARNING_RATE_BOOST = 0.006
MAX_BOOST_EPOCH = 800


def calculate_stepwise_returns(rewards, discount_factor):
    returns = []
    total_rewards = 0

    for reward in reversed(rewards):
        total_rewards = reward + total_rewards * discount_factor
        returns.insert(0, total_rewards)
    returns = torch.tensor(returns)
    normalized_returns = (returns - returns.mean()) / returns.std()
    return normalized_returns


def forward_pass(env, policy, discount_factor):
    log_prob_actions = []
    rewards = []
    done = False
    episode_return = 0

    policy.train()
    observation, info = env.reset()
    prev_observation = observation

    while not done:
        observation = torch.FloatTensor(observation).unsqueeze(0)
        distribution = dist.Categorical(policy(observation))
        action = distribution.sample()
        log_prob_action = distribution.log_prob(action)

        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        if prev_observation[6] and prev_observation[7] and action == 0:
            reward += 0.5 # increase reward for doing nothing after both leg touched ground
        if (prev_observation[6] or prev_observation[7]): # once at least one leg touch ground
            if action == 0:
                reward += 0.2 # increase reward for doing nothing 
            elif action == 2:
                reward -= 0.1 # decrease it for firing main engine
            else:
                reward -= 0.05 # decrease it slighlty for firing side engine
        prev_observation = observation

        log_prob_actions.append(log_prob_action)
        rewards.append(reward)
        episode_return += reward

    log_prob_actions = torch.cat(log_prob_actions)
    stepwise_returns = calculate_stepwise_returns(rewards, discount_factor)

    return episode_return, stepwise_returns, log_prob_actions


def calculate_loss(stepwise_returns, log_prob_actions):
    loss = -(stepwise_returns * log_prob_actions).sum()
    return loss


def update_policy(stepwise_returns, log_prob_actions, optimizer):
    stepwise_returns = stepwise_returns.detach()
    loss = calculate_loss(stepwise_returns, log_prob_actions)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main(): 

    env = gym.make("LunarLander-v3", continuous=False, enable_wind=False, render_mode="human")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    episode_returns = []

    policy = pm.PolicyNetwork(input_dim, HIDDEN_DIM, output_dim, DROPOUT)
#    policy.load_state_dict(torch.load('./policy.pt', weights_only=True))
    policy.train()

    optimizer = opt.Adam(policy.parameters(), lr = LEARNING_RATE)

    for episode in range(1, MAX_EPOCHS+1):
        episode_return, stepwise_returns, log_prob_actions = forward_pass(env, policy, DISCOUNT_FACTOR)

        if episode_return > 0 and episode < MAX_BOOST_EPOCH:
            print(f'Boosting learning rate at episode {episode:3} with score {episode_return:5.1f}!')
            optimizer.param_groups[0]['lr'] = LEARNING_RATE_BOOST 
        _ = update_policy(stepwise_returns, log_prob_actions, optimizer)
        optimizer.param_groups[0]['lr'] = LEARNING_RATE

        episode_returns.append(episode_return)
        mean_episode_return = np.mean(episode_returns[-N_TRIALS:])

        if episode % PRINT_INTERVAL == 0:
            print(f'| Episode: {episode:3} | Mean Rewards: {mean_episode_return:5.1f} |')
        
        if mean_episode_return >= REWARD_THRESHOLD:
            print(f'Reached reward threshold in {episode} episodes')
            break

    # save the learned policy
    for param in policy.state_dict():
        print(param, "\t", policy.state_dict()[param].size())

    torch.save(policy.state_dict(), './policy.pt')

main()

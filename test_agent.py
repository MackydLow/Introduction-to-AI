# Adapted from Gymnasium documentation and https://www.datacamp.com/tutorial/policy-gradient-theorem 

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.distributions as distributions

import numpy as np

import gymnasium as gym

import policy_model as pm


HIDDEN_DIM = 128
DROPOUT = 0.2



def main():

    env = gym.make("LunarLander-v3", continuous=False, enable_wind=False, render_mode="human")

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    policy = pm.PolicyNetwork(input_dim, HIDDEN_DIM, output_dim, DROPOUT)
    policy.load_state_dict(torch.load('./policy-good3.pt', weights_only=True))
    policy.eval()

    episode_scores = []

    for episode in range(20):
        observation, info = env.reset()
        score = 0.0
        done = False

        while not done:
            observation = torch.FloatTensor(observation).unsqueeze(0)
            dist = distributions.Categorical(policy(observation))
            action = dist.sample().item()
            
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
            
        episode_scores.append(score)
        print(f'| Episode: {episode:3} | Score: {score:5.1f} |')

    mean_score = np.mean(episode_scores)
    print(f'Average Score: {mean_score:5.1f}')

    return


if __name__ == "__main__":
    main()

import torch
import torch.optim as opt
import torch.distributions as dis
import numpy as np
import gymnasium as gym

import policyModel as pm
import enviroment as env 
from enviroment import village, houseDelivery, createPriceArray

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

EPS = 1e-8

def calculateStepWise(reward, discountFactor):
    returns = []
    r = 0.0

    for x in reversed(reward):
        r = x + discountFactor * r
        returns.insert(0, r)

    returns = torch.tensor(returns, dtype = torch.float32)

    if returns.std().item() == 0:
        return returns - returns.mean()
    
    return (returns - returns.mean()) / (returns.std() + EPS)

def forwardPass(env, policy, discountFactor):
    logProbAction = []
    rewards = []

    policy.train()
    observation, info = env.reset()
    done = False
    episodeReturn = 0.0

    while not done:
        obsFlat = torch.FloatTensor(observation).view(1, -1)

        probs = policy(obsFlat)
        distr = dis.Categorical(probs)

        action = distr.sample()
        logProb = distr.logProb(action)

        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        logProbAction.append(logProb)
        rewards.append(float(reward))
        episodeReturn += float(reward)

    logProbActions = torch.cat(logProbAction)
    stepwiseReturns = calculateStepWise(rewards, discountFactor)

    return episodeReturn, stepwiseReturns, logProbAction

def calculateLoss(stepwiseReturns, logProbActions):
    return -(stepwiseReturns * logProbActions).sum()

def updatePolicy(stepwiseReturns, logProbAction, optimizer):
    stepwiseReturns = stepwiseReturns.detach()
    loss = calculateLoss(stepwiseReturns, logProbAction)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    print(village)
    

if __name__ == "__main__":
    main()
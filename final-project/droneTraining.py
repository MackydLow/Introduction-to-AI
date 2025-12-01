import torch
import torch.optim as opt
import torch.distributions as dis
import numpy as np
import gymnasium as gym

import policyModel as pm
from policyModel import PolicyModel

import enviroment as env
from enviroment import village, houseDelivery, createPriceArray, villageEnviroment

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
        obsTensor = torch.FloatTensor(observation).unsqueeze(0)

        probs = policy(obsTensor)
        distr = dis.Categorical(probs)

        action = distr.sample()
        logProb = distr.log_prob(action)      

        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        logProbAction.append(logProb)
        rewards.append(float(reward))
        episodeReturn += float(reward)

    logProbActions = torch.cat(logProbAction)
    stepwiseReturns = calculateStepWise(rewards, discountFactor)

    return episodeReturn, stepwiseReturns, logProbActions

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

    print("run")
    deliveryLocation = houseDelivery()
    price = createPriceArray(village)

    env = villageEnviroment(village, price, deliveryLocation = deliveryLocation)

    inputDim = int(np.prod(env.observation_space.shape))
    outputDim = env.actionSpace.n

    policy = PolicyModel(inputDim, HIDDEN_DIM, outputDim, DROPOUT)
    optimizer = opt.Adam(policy.parameters(), lr = LEARNING_RATE)

    episode_returns = []

    for episode in range(1, MAX_EPOCHS + 1):
        episodeReturn, stepwiseReturns, logProbActions = forwardPass(env, policy, DISCOUNT_FACTOR)

        if episode < MAX_BOOST_EPOCH:
            optimizer.param_groups[0]["lr"] = LEARNING_RATE_BOOST
        else:
            optimizer.param_groups[0]["lr"] = LEARNING_RATE

        _ = updatePolicy(stepwiseReturns, logProbActions, optimizer)

        episode_returns.append(episodeReturn)
        mean_return = np.mean(episodeReturn[-N_TRIALS:])

        if episode % PRINT_INTERVAL == 0:
            print(f"| Episode {episode:4} | "
                  f"Mean {N_TRIALS}: {mean_return:6.2f} | " 
                  f"Return: {episodeReturn:6.2f}")

        if mean_return >= REWARD_THRESHOLD:
            print(f"Reached reward threshold at episode {episode}")
            break

    torch.save(policy.state_dict(), "drone_policy.pt")
    print("saved training model -> drone_policy.pt")

if __name__ == "__main__":
    main()
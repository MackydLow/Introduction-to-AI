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

LEARNING_RATE = 0.001
LEARNING_RATE_BOOST = 0.0005
ENTROPY_COEF = 0.01
MAX_BOOST_EPOCH = 800

EPS = 1e-8

def calculateStepWise(reward, discountFactor):
    returns = []
    r = 0.0

    for x in reversed(reward):
        r = x + discountFactor * r
        returns.insert(0, r)

    returns = torch.tensor(returns, dtype = torch.float32)

    if returns.std().item() < EPS:
        return returns - returns.mean()
    else:
        return (returns - returns.mean()) / (returns.std() + EPS)

def forwardPass(env, policy, discountFactor):
    logProbAction = []
    entrop = []
    rewards = []

    policy.train()
    observation, info = env.reset()
    done = False
    episodeReturn = 0.0

    steps = 0
    maxSteps = 300   

    while not done and steps < maxSteps:
        obsTensor = torch.FloatTensor(observation).unsqueeze(0)

        probs = policy(obsTensor)
        distr = dis.Categorical(probs)

        action = distr.sample()
        logProb = distr.log_prob(action)  
        entropy = distr.entropy()    

        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated 

        logProbAction.append(logProb)
        entrop.append(entropy)
        rewards.append(float(reward))
        episodeReturn += float(reward)

        steps += 1

    logProbActions = torch.stack(logProbAction)
    entrop = torch.stack(entrop)
    stepwiseReturns = calculateStepWise(rewards, discountFactor)

    return episodeReturn, stepwiseReturns, logProbActions, entrop

def calculateLoss(stepwiseReturns, logProbActions, entrop):
    policyLoss = -(stepwiseReturns * logProbActions).sum()
    entropLoss = - ENTROPY_COEF * entrop.sum()
    return policyLoss + entropLoss

def updatePolicy(policy, stepwiseReturns, logProbAction, optimizer):
    stepwiseReturns = stepwiseReturns.detach()
    loss = calculateLoss(stepwiseReturns, logProbAction)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return loss.item()

def main():

    print("Starting Training")
    deliveryLocation = houseDelivery()
    price = createPriceArray(village)
    env = villageEnviroment(village, price, deliveryLocation = deliveryLocation)

    inputDim = int(np.prod(env.observation_space.shape))
    outputDim = env.actionSpace.n

    policy = PolicyModel(inputDim, HIDDEN_DIM, outputDim, DROPOUT)
    optimizer = opt.Adam(policy.parameters(), lr = LEARNING_RATE)

    sched = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 300,
        gamma = 0.5
    )

    episode_returns = []

    for episode in range(1, MAX_EPOCHS + 1):
        episodeReturn, stepwiseReturns, logProbActions, entrop = forwardPass(env, policy, DISCOUNT_FACTOR)
        loss = calculateLoss(stepwiseReturns, logProbActions, entrop)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        sched.step()

        episode_returns.append(episodeReturn)
        avgReturn = np.mean(episode_returns[-20:])

        if episode % PRINT_INTERVAL == 0:
            print(f"| Episode {episode:4} | "
                  f"Mean 20: {avgReturn:6.2f} | " 
                  f"Return: {episodeReturn:6.2f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    torch.save(policy.state_dict(), "drone_policy_entropy.pt")
    print("done saved")

if __name__ == "__main__":
    main()

    
#import from liabry
import torch
import torch.optim as opt
import torch.distributions as dis
import numpy as np
import gymnasium as gym

#import policy Model
import policyModel as pm
from policyModel import PolicyModel

#import enviroment and all neccessary functions
import enviroment as env
from enviroment import village, houseDelivery, createPriceArray, villageEnviroment

#setup parameters
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

#calculate discount returns
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

#collect training data
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

    #sample a action and get probability 
    while not done and steps < maxSteps:
        obsTensor = torch.FloatTensor(observation).unsqueeze(0)

        probls = policy(obsTensor)
        distr = dis.Categorical(probls)

        action = distr.sample()
        logProb = distr.log_prob(action)  
        entropy = distr.entropy()    

        #check if episode has ended 
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated 

        #store info 
        logProbAction.append(logProb)
        entrop.append(entropy)
        rewards.append(float(reward))
        episodeReturn += float(reward)

        steps += 1

    logProbActions = torch.stack(logProbAction)
    entrop = torch.stack(entrop)
    stepwiseReturns = calculateStepWise(rewards, discountFactor)

    return episodeReturn, stepwiseReturns, logProbActions, entrop

#computes gradient loss and entropy 
def calculateLoss(stepwiseReturns, logProbActions, entrop):
    adv = stepwiseReturns - (stepwiseReturns.mean())
    policyLoss = -(adv * logProbActions).sum() - 0.01 * entrop.sum()
    entropLoss = - ENTROPY_COEF * entrop.sum()
    return policyLoss + entropLoss

def main():

    #start training
    print("Starting Training")

    #get locations and enviroment set
    deliveryLocation = houseDelivery()
    price = createPriceArray(village)
    env = villageEnviroment(village, price, deliveryLocation = deliveryLocation)

    inputDim = int(np.prod(env.observationSpace.shape))
    outputDim = env.actionSpace.n

    #set up policy
    policy = PolicyModel(inputDim, HIDDEN_DIM, outputDim, DROPOUT)
    optimizer = opt.Adam(policy.parameters(), lr = LEARNING_RATE)

    #set episdoes 
    batchEpisodes = 30

    episodeReturns = []

    #start training loop
    for episode in range(1, MAX_EPOCHS + 1):

        allReturns = []
        allLogProbs = []
        batchReward = []
        allEntrop = []

        for _ in range(batchEpisodes):
            episodeReturn, stepwiseReturns, logProbActions, entrop = forwardPass(env, policy, DISCOUNT_FACTOR)
            allReturns.append(stepwiseReturns)
            allLogProbs.append(logProbActions)
            batchReward.append(episodeReturn)
            allEntrop.append(entrop)
            episodeReturns.append(episodeReturn)

        #reinforce loss
        allReturns = torch.cat(allReturns)
        allLogProbs = torch.cat(allLogProbs)
        allEntrop = torch.cat(allEntrop)

        #reduce variance
        baseline = allReturns.mean()

        #policy gradient 
        adv = allReturns - baseline
        adv = (adv - adv.mean()) /  (adv.std() + 1e-8)

        #calulcate loss
        Polloss = -(adv * allLogProbs).sum()
        entropLoss = -ENTROPY_COEF * allEntrop.sum()

        loss = Polloss + entropLoss

        #optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        #set learning rate decay
        lr = LEARNING_RATE * (0.995 ** episode)
        lr = max(lr, 1e-4)
        optimizer.param_groups[0]["lr"] = lr

        tMean = np.mean(episodeReturns[-N_TRIALS:])
        avgBatchReward = np.mean(batchReward)

        #print progress
        if episode % PRINT_INTERVAL == 0:
            print(f"| Episode {episode:4} | "
                  f"Mean {N_TRIALS}: {tMean:6.2f} | " 
                  f"Return: {avgBatchReward:6.2f} | "
                  f"LR: {lr:.6f}")
            
        if tMean >= REWARD_THRESHOLD:
            print("reached threshold")
            break
            
    torch.save(policy.state_dict(), "drone_policy.pt")
    print("saved -> drone_policy.pt")

if __name__ == "__main__":
    main()
    

    
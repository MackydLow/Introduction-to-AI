import random

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as nnf
import torch.distributions as dist

import numpy as np

import gymnasium as gym  
from gymnasium import spaces



#0 - road (drone can move here, cost: 1)
#1 - green space (drone can move here, cost: 1)
#2 - high wind (drone can move here, cost: 3)
#3 - house (potential drop of zone, drone cant fly here)
#4 - no fly zone (drone cant fly here)
#5 - warehouse (where drone will leave from)

village = [
    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 3, 3, 3, 0, 3, 3, 3],
    [0, 1, 1, 3, 3, 3, 0, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 0, 3, 3, 3, 3, 3, 3],
    [2, 2, 2, 0, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 0, 1, 1, 1, 4, 4, 4],
    [0, 0, 0, 0, 1, 1, 1, 4, 4, 4],
    [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 3, 0, 3, 3, 3, 3, 3, 3]
]

def houseDelivery():
    randomX = random.randint(0, 9)
    randomY = random.randint(0, 9)
    found = False
    while found == False:
        if village[randomX][randomY] == 3:
            found = True
            deliveryLocation = village[randomX][randomY]
            return (randomX, randomY)
        randomX = random.randint(0, 9)
        randomY = random.randint(0, 9)

def createPriceArray(village):
    price = village
    for x in range(len(village)):
        for y in range (len(village)):
            if village[x][y] == 0 or village[x][y] == 1 or village[x][y] == 5:
                price[x][y] = -1
            elif village[x][y] == 2:
                price[x][y] = -3
            elif village[x][y] ==3 or village[x][y] == 4:
                price[x][y] = -20

    return price
            
class villageEnviroment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, village, price, deliveryLocation: tuple):
        super().__init__()

        village = np.array(village)
        price = np.array(price)

        assert village.shape == price.shape
        self.village = village
        self.price = price
        self.rows, self.columns = village.shape

        self.actionSpace = gym.spaces.Discrete(4)

        self.obsv = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (self.rows, self.columns, 1),
            dtype=np.uint8
        )

        self.reset(deliveryLocation)

    def reset(self, deliveryLocation ,seed = None, options = None):
        super().reset(seed=seed)

        deliveryPickup = np.argwhere(self.village == 5)
        
        deliveryPickupRow, deliveryPickupCol = deliveryPickup
        self.dronePos = int[(deliveryPickupRow), int(deliveryPickupCol)]

        deliveryDropOff = deliveryLocation

        dropOffRow, dropOffCol = deliveryDropOff
        self.dropOff = (int(dropOffRow), int(dropOffCol))

        return self.getObs(), {}
    
    def step(self, move):
        x, y = self.dronePos
        nx, ny = x, y

        if move == 0 and x > 0:
            nx -=1
        elif move == 1 and x < self.rows - 1:
            nx += 1
        elif move == 2 and y > 0:
            ny -= 1
        elif move == 3 and y < self.columns - 1:
            ny += 1
        else :
            nx, ny = x,y

        self.dronePos = [x, y]

        reward = float(self.price[nx, ny])
        terminated = False

        if [nx, ny] == self.dropOff:
            reward += 20
            terminated = True

        obs = self.getObs()

        return obs, reward, terminated, False, {}
    
    def getObs(self):
        obs = np.copy(self.village)
        obs[self.dronePos[0], self.dronePos[1]] = 9
        return obs[:, :, None]
    
    def print(self):
        print(self.getObs()[:, :, 0])


def main():
    deliveryLocation = houseDelivery()
    price = createPriceArray(village)

    env = villageEnviroment(village, price, deliveryLocation = deliveryLocation)

    obs, info = env.reset(deliveryLocation)
    print("start grid")
    env.print()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

    print("Action:", action, "Reward:", reward)
    env.print()

    if done:
        print("Taxi Reached")

if __name__ == "__main__":
    main()
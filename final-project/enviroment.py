import random

import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as nnf
import torch.distributions as dist

import numpy as np

import gymnasium as gym  



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

class villageEnviroment(gym.env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, village, price):
        super().__init__()

        self.village = village
        self.H, self.W = village.shape

        self.price = price

        self.observ = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (self.H, self.W, 1),
            dtype=np.uint8
        )

        self.reset()

    def reset(self, seed = None, options = None)
        super().reset(seed=seed)

        deliveryPickup = np.argwhere(self.village == 2)
        self.dronePos = list(deliveryPickup[0])

        houseDropOff = np.argwhere(self.village == 3)
        self.dropOff = tuple(houseDropoff[0])

        return self.get_observation(), {}
    
    def step(self, action):
        x, y = self.dronePos

        if action == 0 and x > 0:
            x -=1
        elif action == 1 and x < self.H - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.W - 1:
            y += 1

        self.dronePos = [x, y]

        reward = self.price
        terminated = False

        if tuple(self.dronePos) == self.dropOff:
            reward += 20
            terminated = True

        return self.get_observation(), reward, terminated, False, {}
    
    def get_observation(self):
        obs = np.copy(self.village)
        obs[self.dronePos[0], self.dronePos[1]] = 9
        return obs[:, :, None]
    
    def render(self):
        print(self.get_observation()[:, :, 0])


def main():
    deliveryLocation = houseDelivery()

if __name__ == "__main__":
    main()
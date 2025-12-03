import random

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
            return (randomX, randomY)
        randomX = random.randint(0, 9)
        randomY = random.randint(0, 9)

def createPriceArray(village):
    price = []
    for x in range(len(village)):
        newPriceRow = []
        for y in range (len(village)):
            if village[x][y] == 0 or village[x][y] == 1 or village[x][y] == 5:
                newPriceRow.append(-1)
            elif village[x][y] == 2:
                newPriceRow.append(-3)
            elif village[x][y] ==3 or village[x][y] == 4:
                newPriceRow.append(-20)
        price.append(newPriceRow)

    return price
            
class villageEnviroment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, village, price, deliveryLocation: tuple):
        super().__init__()

        village = np.array(village)
        price = np.array(price)

        self.village = village
        self.price = price
        self.rows, self.columns = village.shape

        self.actionSpace = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Box(
            low = 0,
            high = np.max(village),
            shape = (self.rows, self.columns, 1),
            dtype=np.float32
        )
        
        self.dronePos = None
        self.dropOff = deliveryLocation 
        self.reset()

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        deliveryPickup = np.argwhere(self.village == 5)
        self.dronePos = tuple(deliveryPickup[0])

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

        self.dronePos = (nx, ny)

        print(self.dronePos, "Drone")

        reward = float(self.price[nx][ny])
        terminated = False

        if (nx, ny) == self.dropOff:
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


##def main():
   # deliveryLocation = houseDelivery()
    #price = createPriceArray(village)

    #env = villageEnviroment(village, price, deliveryLocation = deliveryLocation)

    #bs, info = env.reset(options = {"deliveryLocation": deliveryLocation})
    #print("start grid")
    #env.print()

    #done = False
    #while not done:
        #action = env.actionSpace.sample()
        #obs, reward, done, truncated, info = env.step(action)

    #print("Action:", action, "Reward:", reward)
    #env.print()

    #if done:
        #print("Taxi Reached")

#if __name__ == "__main__":
    #main()
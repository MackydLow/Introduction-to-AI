#imports
import random

import numpy as np
import gymnasium as gym  


#village index
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

#get house that will be delivered to
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

#create array with price for each move
def createPriceArray(village):
    price = []
    for x in range(len(village)):
        newPriceRow = []
        for y in range (len(village)):
            if village[x][y] == 0 or village[x][y] == 1 or village[x][y] == 5:
                newPriceRow.append(-0.1)
            elif village[x][y] == 2:
                newPriceRow.append(-1)
            elif village[x][y] ==3 or village[x][y] == 4:
                newPriceRow.append(-10)
        price.append(newPriceRow)

    return price

#set up gym enviroment            
class villageEnviroment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, village, price, deliveryLocation: tuple):
        super().__init__()

        #convert to numpy
        village = np.array(village)
        price = np.array(price)

        self.village = village
        self.price = price
        self.row, self.column = village.shape

        #set up all 4 possible moves 
        self.actionSpace = gym.spaces.Discrete(4)

        #set up obervation space
        self.observationSpace = gym.spaces.Box(
            low = 0,
            high = np.max(village),
            shape = (self.row, self.column, 1),
            dtype=np.float32
        )
        #initiliase drone and drop off
        self.droneLoc = None
        self.dropOff = deliveryLocation 
        self.dropOff = (deliveryLocation[0], deliveryLocation[1])
        self.reset()

    #reset to starting state
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        #find pickup area
        deliveryPickup = np.argwhere(self.village == 5)
        self.droneloc = tuple(deliveryPickup[0])

        return self.getObs(), {}
    
    #move the drone
    def step(self, move):
        x, y = self.droneLoc
        nx, ny = x, y

        #move up
        if move == 0 and x > 0:
            nx -=1
        #move down
        elif move == 1 and x < self.row - 1:
            nx += 1
        #move left
        elif move == 2 and y > 0:
            ny -= 1
        #move right
        elif move == 3 and y < self.column - 1:
            ny += 1
        else :
            nx, ny = x,y

        #calculate distance improvement
        oldDistance = abs(x - self.dropOff[0]) + abs(y-self.dropOff[1])
        newDistance = abs(nx - self.dropOff[0]) + abs(ny-self.dropOff[1])

        #update position
        self.droneLoc = (nx, ny)

        #penalties for improved results
        reward = float(self.price[nx][ny]) * 0.01
        reward -= 0.01

        reward += (oldDistance - newDistance) * 0.5

        terminated = False

        #delivery complete and reward
        if (nx, ny) == self.dropOff:
            reward += 75
            terminated = True

        obs = self.getObs()

        return obs, reward, terminated, False, {}
    
    #build obervation grid
    def getObs(self):
        obs = np.copy(self.village)
        obs[self.droneLoc[0], self.droneLoc[1]] = 9
        return (obs[:, :, None].astype(np.float32)) / 4
    
    def print(self):
        print(self.getObs()[:, :, 0])

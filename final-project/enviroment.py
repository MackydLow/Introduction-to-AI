import random

#0 - road (drone can move here, cost: 1)
#1 - green space (drone can move here, cost: 1)
#2 - water (drone can move here, cost: 3)
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

def main():
    deliveryLocation = houseDelivery()

if __name__ == "__main__":
    main()
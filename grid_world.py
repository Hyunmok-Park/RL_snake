import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class myenv():
    def __init__(self, height=10, width=10):
        self.height = height
        self.width = width
        self.x = 0
        self.y = 0
        self.dx = [1, 0, -1, 0]
        self.dy = [0, 1, 0, -1]

        self.food_x = None
        self.food_y = None

        self.create_food()

    def reset(self):
        self.x = 0
        self.y = 0
        self.create_food()
        return self.x, self.y, self.food_x, self.food_y

    def step(self, action):

        if action == 0: #right
            if self.x == (self.width - 1):
                pass
            else:
                self.x = self.x + 1
        elif action == 1: #up
            if self.y == 0:
                pass
            else:
                self.y = self.y - 1
        elif action == 2: #left
            if self.x == 0:
                pass
            else:
                self.x = self.x - 1
        elif action == 3 : #down
            if self.y == (self.height - 1):
                pass
            else:
                self.y = self.y + 1

        if self.x == self.food_x and self.y == self.food_y:
            reward = 1
            done = True
        else:
            done = False
            reward = -1

        return (self.x, self.y, self.food_x, self.food_y), reward, done

    def create_food(self):
        done = False
        while not done:
            x = np.random.choice([i for i in range(self.width)])
            y = np.random.choice([i for i in range(self.height)])

            if x == self.food_x and y == self.food_y:
                done = False
            else:
                done = True

        self.food_x = x
        self.food_y = y

    def draw_world(self):
        return 0


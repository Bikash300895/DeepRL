# author: @bikash.shuvendu

import numpy as np
import matplotlib.pyplot as plt

""" >> Defining the class """"
class Grid:
    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        # rewards should be dict of: (i, j): r(row, col): reward
        # actions should bt dict of: (i, j): A(row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -=1
            elif action == 'D':
                self.i +=1
            elif action == 'R':
                self.j +=1
            elif action == 'L':
                self.j -=1
        # return a reward if any
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        if action == 'U':
            self.i +=1
        elif action == 'D':
            self.i -=1
        elif action == 'R':
            self.j -=1
        elif action == 'L':
            self.j +=1
        assert(self.current_state() in self.all_stated())

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(self.action.keys() + self.rewards.keys())

    def standard_grid():
        # define a grid that desctibes the reward for arriving at each set_state
        # the possible action at each set_state
        g = Grid()
        rewards = {(0,3): 1, (1,3): -1}     # simulating the most common grid with +1 at top most corner and -1 just bottom of it
        actions = {
            (0, 0): ('D', 'R'),
            (0, 1): ('L', 'R'),
            (0, 2): ('L', 'D', 'R'),
            (1, 0): ('U', 'D'),
            (1, 2): ('U', 'D', 'R'),
            (2, 0): ('U', 'R'),
            (2, 1): ('L', 'R'),
            (2, 2): ('L', 'R', 'U'),
            (2, 3): ('L', 'U'),
        }
        g.set(rewards, actions)
        return g

    def play_game(agent, env):
        pass

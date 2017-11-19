# author: @ShuvenduBikash

import numpy as np
import matplotlib.pyplot as plt


# Defining the class
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
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        # return a reward if any
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        assert (self.current_state() in self.all_stated())

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_grid():
    # define a grid that desctibes the reward for arriving at each set_state
    # the possible action at each set_state
    g = Grid(3, 4, (0, 0))
    rewards = {(0, 3): 1,
               (1, 3): -1}  # simulating the most common grid with +1 at top most corner and -1 just bottom of it
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


def negative_grid(step_cost=-0.1):
    # in this game we want to try to minimize the number of moves
    # so we will penalize every move
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g


# helper function for visualization the training
def print_values(V, g):
    for i in range(g.width):
        print("\n-------------------------")
        for j in range(g.height):
            v = V.get((i, j), 0)

            if v >= 0:
                print(" %.2f" % v, end=" ")
            else:
                print("%.2f" % v, end=" ")  # -ve sign takes an extra space


def print_policy(P, g):
    for i in range(g.width):
        print("\n-------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end=" ")


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val

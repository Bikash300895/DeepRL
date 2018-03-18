import numpy as np
import matplotlib.pyplot as plt


def print_values(V, g):
    print("\nValues: ")
    for i in range(g.width):
        print("---------------------------")
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")  # -ve sign takes up an extra space
        print("")


def print_policy(P, g):
    print("\nPolicy: ")
    for i in range(g.width):
        print("---------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end="")
        print("")


# Environment class
class Grid:
    """
    Contain the grid world environment
    :param
    width
    height
    (i, j) : location
    rewards: dictionary
    actions: dictionary
    """

    def __init__(self, width, height, start):
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        """ function for setting the value of reward and actions
        rewards and actions are dict with index of tuple(i, j) the grid location """
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        """ Set the agent in new location s """
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions  # if there is not actions correspondin to s, then it is a terminal

    def move(self, action):
        """ Move if it is valid move, and return the associated reward """
        # check if legal move
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1

            elif action == 'D':
                self.i += 1

            elif action == 'R':
                self.j += 1

            elif action == 'L':
                self.j -= 1

        return self.rewards.get((self.i, self.j), 0)  # 0 is the default reward if nothing is assigned

    def undo_mode(self, action):
        if action == 'U':
            self.i += 1

        elif action == 'D':
            self.i -= 1

        elif action == 'R':
            self.j -= 1

        elif action == 'L':
            self.j += 1

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


def standard_grid():
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
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

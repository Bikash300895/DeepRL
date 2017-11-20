# Author: @ShuvenduBikash
import numpy as np
import matplotlib.pyplot as plt
from grid_world import *

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')
LEARNING_RATE = 0.001

if __name__ == '__main__':
    grid = standard_grid()

    print("Rewards: ")
    print_values(grid.rewards, grid)

    # state -> action
    # found by policy_iteration_random on standard_grid
    # MC method won't get exactly this, but should be close
    # values:
    # ---------------------------
    #  0.43|  0.56|  0.72|  0.00|
    # ---------------------------
    #  0.33|  0.00|  0.21|  0.00|
    # ---------------------------
    #  0.25|  0.18|  0.11| -0.17|
    # policy:
    # ---------------------------
    #   R  |   R  |   R  |      |
    # ---------------------------
    #   U  |      |   U  |      |
    # ---------------------------
    #   U  |   L  |   U  |   L  |
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }

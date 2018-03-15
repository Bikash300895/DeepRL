"""
Iterative policy evaluation ::
given a policy pi, return V

Steps:
    1. Initialize V : Value of all states
    2. Repeat until converge:
        calculate V(s) and subtract from previous time step

"""

import numpy as np
from grid_world import standard_grid
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Initialize local variables
    grid = standard_grid()
    states = grid.all_states()
    gamma = 1.0
    small_enough = 1e-4

    # Initialize the state-values V
    V = {}
    for s in states:
        V[s] = 0


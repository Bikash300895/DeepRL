"""
value iteration process eliminate two iterative process with one
it tries to optimize the process directly then take argmax to find the optimal policy
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import negative_grid, standard_grid
from iterative_policy_evaluation import print_policy, print_values

if __name__ == '__main__':
    grid = negative_grid()
    gamma = 0.9

    # initialize state-value V(s)
    V = {}
    for s in grid.all_states():
        V[s] = 0

    # initialize policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = grid.actions[s][0]

    # Value iteration
    states = grid.actions.keys()
    while True:
        value_converged = True
        for s in states:
            old_v = V[s]
            for a in grid.actions[s]:
                grid.set_state(s)
                r = grid.move(a)
                v = r + gamma * V[grid.current_state()]

                if v > V[s]:
                    V[s] = v
                    policy[s] = a
                    value_converged = False

        if value_converged:
            break

    print_values(V, grid)
    print_policy(policy, grid)

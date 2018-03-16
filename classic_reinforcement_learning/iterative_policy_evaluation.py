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

    while True:
        biggest_change = 0
        for s in grid.actions:
            v = V[s]
            new_v = 0
            prob = 1 / (len(grid.actions[s]))
            for a in grid.actions[s]:
                grid.set_state(s)
                r = grid.move(a)
                new_v += prob * (r + gamma * V[grid.current_state()])
            V[s] = new_v
            biggest_change = max(biggest_change, np.abs(v - V[s]))

        if biggest_change < small_enough:
            break

    print("Values for uniformly random actions: ")
    print_values(V, grid)

    ### fixed policy ###
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    print_policy(policy, grid)

    # initialize V(s)
    V = {}
    for s in states:
        V[s] = 0
    gamma = 0.9

    while True:
        biggest_change = 0
        for s in states:
            if s in policy:
                old_v = V[s]

                grid.set_state(s)
                r = grid.move(policy[s])

                V[s] = r + gamma * V[grid.current_state()]
                biggest_change = np.abs(old_v - V[s])

        if biggest_change < small_enough:
            break

    print_values(V, grid)

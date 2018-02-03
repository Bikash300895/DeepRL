import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

small_enough = 1e-3  # threshold for convergence


def print_values(V, g):
    for i in range(g.width):
        print("----------------------------")
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v >= 0:
                print("  %.2f|" % v, end="")
            else:
                print(" %.2f|" % v, end="")  # -ve take an extra space
        print()


def print_policy(P, g):
    for i in range(g.width):
        print("----------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  %s  |" % a)
        print()


if __name__ == '__main__':
    grid = standard_grid()
    states = grid.all_states()

    # assign uniformly random action
    V = {}

    for s in states:
        V[s] = 0
        gamma = 1.0  # discount factor

    # repeat until converge
    while True:
        biggest_change = 0

        # go over all states
        for s in states:
            old_v = V[s]

            # if there is action for this state (not terminal state)
            if s in grid.actions:
                new_v = 0
                p_a = 1.0 / len(grid.actions[s])  # giving equal possibility

                # sum up the returns
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])

                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        if biggest_change < small_enough:
            break

    print('values for uniform search')
    print_values(V, grid)

    """
    values for uniform search
    ----------------------------
     -0.03|  0.09|  0.22|  0.00|
    ----------------------------
     -0.16|  0.00| -0.44|  0.00|
    ----------------------------
     -0.29| -0.41| -0.54| -0.77|
    """
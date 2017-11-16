# author: @ShuvenduBikash

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

# threshold convergence
SMALL_ENOUGH = 10e-4


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


def print_poicy(P, g):
    for i in range(g.width):
        print("\n-------------------------")
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print("  %s  |" % a, end=" ")


if __name__ == '__main__':
    # iterative policy evaluation
    # given a policy, let's fidn it's value fucntion V(s)
    # NOTE:
    # there are 2 sources of randomness
    # p(a|s) - decising what action to take given the set_state
    # p(s', r|s, a) - the next state and reward given your action-state pair

    grid = standard_grid()

    states = grid.all_states()

    """Uniformly random actions"""
    # initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0

    # discount factor
    gamma = 1.0

    # reapeat until convergence
    while True:
        biggest_chage = 0

        for s in states:
            old_v = V[s]

            # V(s) only has values if it's not terminal states
            if s in grid.actions:
                new_v = 0
                p_a = 1.0 / len(grid.actions[s])

                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)

                    # calculate the new value (the heart of the algorithm)
                    # Equation 4.4
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                biggest_chage = max(biggest_chage, np.abs(old_v - V[s]))
        if biggest_chage < SMALL_ENOUGH:
            break
    print("\nValues for uniformly random actions: ")
    print_values(V, grid)

    """Fixed policy"""
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
    print_poicy(policy, grid)

    # initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0

    # discount factor
    gamma = 0.9

    # reapeat until convergence
    while True:
        biggest_chage = 0

        for s in states:
            old_v = V[s]

            # V(s) only has values if it's not terminal states
            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)

                # calculate the new value (the heart of the algorithm)
                V[s] = (r + gamma * V[grid.current_state()])
                biggest_chage = max(biggest_chage, np.abs(old_v - V[s]))
            # devug printing
            print("\n\n\nApplying next policy")
            print_values(V, grid)

        if biggest_chage < SMALL_ENOUGH:
            break
    print("\nValues for uniformly random actions: ")
    print_values(V, grid)

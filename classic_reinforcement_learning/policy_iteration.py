"""
algorithm for finding optimal policy
strategy :  1. evaluate the policy
            2.  - for each state find the best action
                - if not converge go to step 2
"""
import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

small_enough = 1e-3
gamma = 0.9
ALL_POSSIBLE_ACTIONS =  ('U', 'D', 'L', 'R')

# this is determinstic,
# all p(s',r |s,a) = 1/0

if __name__ == '__main__':
    grid = negative_grid()
    print("Rewards: ")
    print_values(grid.rewards, grid)

    # Initialize with random policy to each state
    policy = {}  # state -> action
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    print("Initaial random policy: ")
    print_policy(policy, grid)

    # Initialize state-values
    V = {}
    states = grid.all_states()
    for s in states:
        # V[s] = 0  # it's ok to initialize all to 0
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            # terminal state
            V[s] = 0

    while True:
        # policy evaluation step
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]

                # Check if not terminal state
                if s in policy:
                    a = policy[s]
                    grid.set_state(s)
                    r = grid.move(a)
                    V[s] = r + gamma * V[grid.current_state()]
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))

            if biggest_change < small_enough:
                break

        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')

                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + gamma * V[grid.current_state()]

                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print("\nValues: ")
    print_values(V, grid)

    print("\nPolicy")
    print_policy(policy, grid)


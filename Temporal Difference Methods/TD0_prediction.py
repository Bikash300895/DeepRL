import numpy as np
import matplotlib.pyplot as plt
from grid_world import *

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps=0.1):
    # we'll use epsilon-soft to ensure all states are visited
    # what happens if you don't do this? i.e. eps=0
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)



def play_game(grid, policy):
    # returns a list of states and corresponding rewards (not returns as in MC)
    s = (2, 0)
    grid.set_state(s)
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards


if __name__ == '__main__':
    grid = standard_grid()

    # print rewards
    print("Rewards: ")
    print_values(grid.rewards, grid)

    # state -> action
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

    # Initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    # repeat
    for it in range(1000):
        states_and_rewards = play_game(grid, policy)

        for t in range(len(states_and_rewards) -1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t+1]

            # now we will update the V(s)
            V[s] = V[s] + ALPHA * (r + GAMMA*V[s2] - V[s])

    print("\nValues: ")
    print_values(V, grid)
    print("\nPolicy: ")
    print_policy(policy, grid)

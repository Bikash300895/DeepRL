import numpy as np
import matplotlib.pyplot as plt
from classic_reinforcement_learning.grid_world import negative_grid, standard_grid
from classic_reinforcement_learning.iterative_policy_evaluation import print_values, print_policy

GAMMA = 0.9
ALL_POSSIBLE_ACTION = ('U', 'D', 'L', 'R')


def play_game(grid, policy):
    # returns a list of states and corresponding returns

    # Initialize: choose the start state and first action randomly
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))  # int
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTION)

    # Play and episode and store the rewards
    states_action_rewards = [(s, a, 0)]  # r(t) results for taking action a(t-1) at s(t-1)
    while True:
        old_s = grid.current_state()
        r = grid.move(a)
        s = grid.current_state()

        if old_s == s:
            # hack: stop agent from bumping into wall and never end episode
            states_action_rewards.append((s, None, -100))
            break
        elif grid.game_over():
            states_action_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_action_rewards.append((s, a, r))

    # calculate the returns by working backward from terminal state
    G = 0
    states_action_returns = []
    first = True
    for s, a, r in reversed(states_action_rewards):
        if first:
            first = False
        else:
            states_action_returns.append((s, a, G))
        G = r + GAMMA * G

    states_action_returns.reverse()
    return states_action_returns


def max_dict(d):
    """
    helper function that find the max(value) and it's key
    :param d: dictionary
    :return: max_key, max_val
    """
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


if __name__ == '__main__':
    grid = negative_grid()
    print("Rewards: ")
    print_values(grid.rewards, grid)

    # initialize random policy
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTION)

    # initialize Q and returns
    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:  # non terminal state
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTION:
                Q[s][a] = 0
                returns[(s, a)] = []
        else:
            # terminal state
            pass

    deltas = []
    for t in range(2000):
        if t % 500 == 0:
            print(t)

        # generate an episode using pi
        biggest_change = 0
        states_action_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_action_returns:
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)

        # update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]

    plt.plot(deltas)
    plt.show()

    print("\nFinal Policy")
    print_policy(policy, grid)

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]

    print("\nFinal values: ")
    print_values(V, grid)
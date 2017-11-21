import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0


def play_one_episode(params, render=False):
    observation = env.reset()
    for i in range(1000):
        if render:
            env.render()

        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)

        if done:
            return i
    return 1000


def play_multiple_episode(T, params, render=False):
    episode_lengths = np.empty(T)

    for i in range(T):
        episode_lengths[i] = play_one_episode(params, render)

    avg_length = episode_lengths.mean()
    print("Average length: ", avg_length)
    return avg_length


def random_search():
    episode_lengths = []
    best = 0
    params = None

    for t in range(100):
        # choose random parameters
        new_params = np.random.random(4)*2 -1
        avg_length = play_multiple_episode(100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length

    return episode_lengths, params


if __name__ == '__main__':
    episode_lengths, params = random_search()
    plt.plot(episode_lengths)
    plt.show()

    print("Final run with best parameters")
    play_multiple_episode(10, params, True)

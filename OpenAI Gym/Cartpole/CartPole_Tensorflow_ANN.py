import random
from collections import Counter

import numpy as np
import gym

env = gym.make('CartPole-v0')
initial_games = 1000
score_requirement = 50
goal_steps = 500


def some_random_games_first():
    for i_episode in range(20):
        # resetting the enc and getting a random initial state
        observation = env.reset()

        # taking maximum of 100 steps if note gameover mean time
        for t in range(goal_steps):
            env.render()
            print(observation)

            # selecting a random action
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} fimesteps".format(t + 1))
                break


# its purpose is to generate data by taking random action that will be use to train the nn model
def initial_population():
    # [obs, moves]
    training_data = []

    # all scores
    scores = []

    # scores that met our threshold
    accepted_score = []

    for _ in range(initial_games):
        env.reset()

        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            # choose random action (0, 1)
            action = random.randrange(0,2)
            # do it
            observation, reward, done, info = env.step(action)

            # let's store  which action, at which observation result in what reward to train the nn
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation  # update observation
            score += reward
            if done: break

        if score>score_requirement:
            accepted_score.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])

        scores.append(score)

    # save to a file
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:', np.mean(accepted_score))
    print('Median score for accepted scores:', np.median(accepted_score))
    print(Counter(accepted_score))

    return training_data

# initial_population()

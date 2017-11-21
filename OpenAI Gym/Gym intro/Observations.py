import gym
env = gym.make('CartPole-v0')

# Staring the episode
for i_episode in range(20):
    # resetting the enc and getting a random initial state
    observation = env.reset()

    # taking maximum of 100 steps if note gameover mean time
    for t in range(100):
        env.render()
        print(observation)

        # selecting a random action
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} fimesteps".format(t+1))
            break
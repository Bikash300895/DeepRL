import gym
env = gym.make('CartPole-v0')

for episode in range(20):
    # playing an episode
    observation = env.reset()
    for i in range(250):
        # print(observation)
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode end at iteration " + str(i))
            break

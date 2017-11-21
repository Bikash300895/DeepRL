import gym

# loading the environemnt
env = gym.make('CartPole-v0')

# putting the agent in the start state
env.reset()

# getting some information
print(env.observation_space)
print(env.action_space)

done = False
counter = 0
while not done:
    counter += 1
    observation, reward, done, _ = env.step(env.action_space.sample())
    
print(counter)
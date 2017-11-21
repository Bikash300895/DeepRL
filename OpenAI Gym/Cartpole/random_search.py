import gym
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0
    
    while not done and t<10000:
        # env.render()
        t += 1
        
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        if done:
            break
    return t



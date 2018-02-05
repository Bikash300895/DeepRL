import numpy as np
import keras
import random
import gym
from keras import layers, models
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factro
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    
    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        model.summary()
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # First dimension is the batch dim


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n  # 2
    agent = DQNAgent(state_size, action_size)
    
    
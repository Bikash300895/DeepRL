# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras import backend as K

eposodes = 5000


class DQNagent:
    """
    Model for playing with Deep Q learning
    """
    def __init__(self, state_size, action_size):
        """
        Arguments:
            state_size: 
                input size to the network, indicating state 
            action_size:
                possible actions
                (for CartPole-v0 it has two actions 0/1)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # store prev states
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        
        
    def _build_model(self):
        # Build the neural network 
        model = Sequential()
        model.add(layers.Dense(24, input_dim = self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # Store outputs in memory for use in future training
        self.memory.append((state, ation, reward, next_state, done))
        
    
        
        
        
        
        
            
                
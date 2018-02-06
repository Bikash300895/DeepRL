import numpy as np
import keras
import random
import gym
from keras import layers, models
from collections import deque

EPISODES = 1000


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
    
    def reply(self, batch_size = 32):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
                
            target_f = self.model.predict(state)  # it will the the output of the model, but we know one (action, reward) pair
            target_f[0][action] = target  # 0 for batch dim, action is the known action. updating the return for that action
            
            self.model.fit(state, target_f, epochs = 1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n  # 2
    agent = DQNAgent(state_size, action_size)
    
    # helper parameters
    done = False
    batch_size = 32
    
    for e in range(EPISODES):
        state = env.reset()  # shape = (4,)
        state = np.reshape(state, [1, state_size])  # shape (1, 4) == (batch_dim, state_size)
        
        for time in range(500):
            # env.render()  # to render the cartpole
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
                
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
        
        if len(agent.memory) > batch_size:
            agent.reply(batch_size)
            
    agent.model.save_weights('CartPole-DQN.h5')
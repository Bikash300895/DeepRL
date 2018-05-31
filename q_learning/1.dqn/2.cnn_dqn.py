import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from IPython.display import clear_output
from utils.wrappers import make_atari, wrap_deepmind, wrap_pytorch

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autograd.Variable(*args, **kwargs)


# helper functions
def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


# reply buffer
from collections import deque


class ReplyBuffer(object):
    def __init__(self, capacity):
        """stores (state, action, reward, next_state, done) in a buffer"""
        self.buffer = deque(maxlen=capacity)                                   # (state, action, reward, next_state, done)  -> tuple

    def push(self, state, action, reward, next_state, done):
        """
        state        ->   (1, 80, 80)
        action       ->   int
        reward       ->   float
        next_state   ->   (1, 80, 80)
        done         ->   bool
        """
        state = np.expand_dims(state, 0)  # insert a batch_dim                 # (1, 1, 80, 80)
        next_state = np.expand_dims(next_state, 0)                             # (1, 1, 80, 80)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))     # all are (32) -> tuple
        return np.concatenate(state), action, reward, np.concatenate(next_state), done             # state -> (32, 1, 80, 80)

    def __len__(self):
        return len(self.buffer)


class CnnDQN(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
        state -> (1, 80, 80)
        
        return action -> int
        """
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0))        # (1, 1, 80, 80) -> numpy
            q_value = self.forward(state)                                      # (1, 6)         -> torch.FloatTensor()
            action = q_value.max(1)[1].item()                                  # max returns  (item, index)
        else:
            action = random.randrange(self.num_actions)

        return action                                                          # int


if __name__ == '__main__':
    env    = make_atari('Pong-v0')
    env    = wrap_deepmind(env)
    env    = wrap_pytorch(env)
    
    state = env.reset()  # (1, 80, 80)


    # epsilon greedy exploration parameters
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    num_frames = 1400000
    batch_size = 32
    gamma = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    # define model
    model = CnnDQN(env.observation_space.shape, env.action_space.n)

    if USE_CUDA:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    replay_initial = 10000
    replay_buffer = ReplyBuffer(100000)

    # compute temporal difference loss
    def compute_td_loss(batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))

        with torch.no_grad():
            next_state = Variable(torch.FloatTensor(np.float32(next_state)))

        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = model(state)
        next_q_values = model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


    # train the model
    episode = 0
    for frame_idx in tqdm(range(1, num_frames + 1)):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        # next_state = np.rollaxis(next_state, 2, 0)

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            print('\tEpisode : %d, Current reward: %d' % (episode, episode_reward))
            episode += 1

            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())

        if frame_idx % 200 == 0:
            pass

    plot(frame_idx, all_rewards, losses)

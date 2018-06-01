import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autorgrad
from IPython.display import clear_output

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autorgrad.Variable(*args, **kwargs).cuda() \
    if USE_CUDA else autorgrad.Variable(*args, **kwargs)


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
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)  # insert a batch_dim
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)

        return action


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state = env.reset()

    # epsilon greedy exploration parameters
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay)

    num_frames = 10000
    batch_size = 32
    gamma = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    # define model
    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model = DQN(env.observation_space.shape[0], env.action_space.n)
    if USE_CUDA:
        current_model = current_model.cuda()
        target_model = target_model.cuda()

    optimizer = optim.Adam(current_model.parameters())
    replay_buffer = ReplyBuffer(1000)
    
    # function for synchoronize current net with target net
    def update_target(current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    # compute temporal difference loss
    def compute_td_loss(batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))

        with torch.no_grad():
            next_state = Variable(torch.FloatTensor(np.float32(next_state)))

        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = current_model(state)
        next_q_values = current_model(next_state)
        next_q_state_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


    # train the model
    episode = 0
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            print('Episode : %d, Current reward: %d' % (episode, episode_reward))
            episode += 1

            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())

        if frame_idx % 200 == 0:
            update_target(current_model, target_model)

    plot(frame_idx, all_rewards, losses)

import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


# Create the architecture of the neural network
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = F.linear(input_size, 30)  # 30 in the number of hidden layer
        self.fc2 = F.linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# implement the experience reply
class ReplyMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        """
        :param event: tuple (state, next_state, action, reward)
        :return: None
        """
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        """
        samples from the stored data
        :param batch_size: size of batch
        :return: a batch of data
        """
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

        # samples = zip(*random.sample(memory, batch_size))
        # samples = map(lambda x: Variable(torch.from_numpy(np.array(x))), samples)
        # print(list(samples))


# Deep Q learning
class Dqn:
    def __init__(self, input_size, nb_actions, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_actions)
        self.memory = ReplyMemory(100000)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # one extra dimension for batch
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 7)  # temperature = 7; it will increase the probability of high prob higher
        action = probs.multinomial()
        return action.data[0, 0]

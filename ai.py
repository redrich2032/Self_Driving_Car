# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        #connection from input layer to hidden layer
        self.fc1 = nn.Linear(input_size, 30)
        #connection from hidden layer to output layer
        self.fc2 = nn.Linear(30, nb_action)
        
        #activate neurons
        #state -> our inputs from input layer
    def forward(self, state):
        #activation function for hidden neurons
        hidden_neurons = F.relu(self.fc1(state))
        q_values = self.fc2(hidden_neurons)
        return q_values

#Implement Experience Replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    
        

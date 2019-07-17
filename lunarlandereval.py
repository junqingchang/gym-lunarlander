import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('LunarLander-v2')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
from IPython import display

MODEL_DIR = 'models'

MODEL_POLICY_NET = os.listdir(MODEL_DIR)

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_shape, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, outputs)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def select_action(state, steps_done, n_actions, policy_net, EPS_START, EPS_END, EPS_DECAY, device):
    steps_done += 1
    outputs = []
    with torch.no_grad():
        for model in policy_net:
            out = model(state.type(torch.FloatTensor).to(device)).max(1)[1].view(1,1)
            outputs.append(out)
        outputs = torch.Tensor(outputs).type(torch.LongTensor)
        return torch.mode(outputs).values, steps_done

def top_models(k, MODEL_POLICY_NET, kinematics_shape, n_actions, device):
    scores = []
    for model in MODEL_POLICY_NET:
        score = float(model[21:-3])
        if score not in scores:
            scores.append(score)
    scores.sort()
    scores.reverse()
    policy_model_set = []
    for i in range(k):
        policy_net = DQN(kinematics_shape, n_actions).to(device)
        policy_net.load_state_dict(torch.load('{}/policy-episode-reward{}.pt'.format(MODEL_DIR,scores[i]), map_location=device))
        policy_net.eval()
        policy_model_set.append(policy_net)
    return policy_model_set

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    env.reset()

    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200

    init_kinematics = torch.Tensor([env.reset()]).to(device)
    kinematics_shape = init_kinematics.shape[1]
    
    n_actions = env.action_space.n

    policy_net = top_models(5, MODEL_POLICY_NET, kinematics_shape, n_actions, device)

    memory = ReplayMemory(200000, Transition)

    steps_done = 0

    # Initialize the environment and state
    env.reset()
    state = torch.Tensor([env.reset()]).to(device)
    total_reward = 0
    for t in count():
        # Select and perform an action
        action, steps_done = select_action(state, steps_done, n_actions, policy_net, EPS_START, EPS_END, EPS_DECAY, device)
        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward

        total_reward = torch.tensor([total_reward], device=device).type(torch.float32)

    
        if not done:
            next_state = torch.Tensor([next_state]).to(device)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, total_reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        env.render()

        if done:
            break


    print('Complete, Reward for run: {}'.format(total_reward[0]))
    env.close()

if __name__ == '__main__':
    main()
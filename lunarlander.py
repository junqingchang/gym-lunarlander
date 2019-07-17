import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('LunarLander-v2')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
from IPython import display

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
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.type(torch.FloatTensor).to(device)).max(1)[1].view(1,1), steps_done
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done

def optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, Transition, device):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def plot_rewards(episode_rewards, is_ipython, display):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def main():
    
    plt.ion()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))

    

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 30
    learning_rate = 5e-4
    weight_decay = 5e-5

    init_kinematics = torch.Tensor([env.reset()]).to(device)
    kinematics_shape = init_kinematics.shape[1]
    
    n_actions = env.action_space.n

    policy_net = DQN(kinematics_shape, n_actions).to(device)
    target_net = DQN(kinematics_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    memory = ReplayMemory(200000, Transition)

    steps_done = 0

    episode_rewards = []

    num_episodes = 2000
    for i_episode in range(num_episodes):
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
            optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE, GAMMA, Transition, device)
            # env.render()

            if done:
                episode_rewards.append(total_reward)
                # plot_rewards(episode_rewards, is_ipython, display)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print('Episode {}: {}'.format(i_episode, total_reward[0]))
        if i_episode % 100 == 0 and i_episode != 0:
            rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            torch.save(policy_net.state_dict(), 'policy-episode-reward{}.pt'.format(means[len(means)-1]))
            torch.save(target_net.state_dict(), 'target-episode-reward{}.pt'.format(means[len(means)-1]))
        
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    torch.save(policy_net.state_dict(), 'policy-episode-reward{}.pt'.format(means[len(means)-1]))
    torch.save(target_net.state_dict(), 'target-episode-reward{}.pt'.format(means[len(means)-1]))
    print('Complete')
    # env.render()
    env.close()
    plt.ioff()
    # plt.show()
    plot_rewards(episode_rewards, is_ipython, display)
    plt.savefig('lunarlander.png')

if __name__ == '__main__':
    main()

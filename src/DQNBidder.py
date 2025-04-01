import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Bidder import Bidder 
import numpy as np

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        # self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(64, n_actions)
        # nn.init.xavier_normal_(self.layer1.weight)
        # nn.init.xavier_normal_(self.layer3.weight)
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        return self.layer3(x)
    





class DQNBidder(Bidder):
    def __init__(self, rng, value):
        super(DQNBidder, self).__init__(rng)
        self.truthful = True
        self.batch_size = 1#@28
        self.gamma = 0.25
        self.eps_start = 0.25
        self.esp_end = 0.05
        self.esp_decay = 0.00002
        self.tau = 1.0#0.5
        self.lr = 1e-3
        self.action_size = value + 1
        self.n_basis = 3
        self.policy_net = DQN(2*self.n_basis, self.action_size).to(device)
        self.target_net = DQN(2*self.n_basis, self.action_size).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(1)#000

        self.steps_done = 0
        self.state = torch.tensor(np.zeros(2*self.n_basis, dtype=np.float32)).to(device)
        values = torch.arange(0, self.action_size, dtype=torch.float32)
        
        self.all_states = torch.tensor(self.state_features(np.array(torch.cartesian_prod(values, values)))).to(device)
        print(self.all_states)
        self.mu = 1.25
        self.a = 2
        self.a0 = 0
        
        # Size of the state feature vector
        # including previous bid, previous first price, previous second price, and action
        
    def state_features(self, state):
        # fourier basis of n degrees
        state = state/self.action_size
        state_features = np.concatenate([np.cos(i * np.pi * state) for i in range(self.n_basis)], axis=-1)
        # print(state_features)
        return state_features
        
    def conditional_argmax(self):
        conditional_argmax = torch.argmax(self.policy_net(self.all_states), dim=1).cpu().detach().numpy()
        # print(conditional_argmax)
        return conditional_argmax
        
    def demand(self, p):
        """Computes demand"""
        e = np.exp((self.a - p) / self.mu)
        d = e / (np.sum(e) + np.exp(self.a0 / self.mu))
        return d

    def convert_prices(self, state):
        pmax = 2.0
        pmin = 1.5
        return state * ((pmax-pmin)/self.action_size) + pmin

    def q_values(self, state, actions):
        return self.policy_net(self.state)[actions].cpu().detach().numpy()

    def bid(self, value, context, estimated_CTR=1):
        sample = random.random()
        eps_threshold = self.esp_end + (self.eps_start - self.esp_end) * \
            math.exp(-self.steps_done * self.esp_decay)
        self.steps_done += 1
        with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.argmax = torch.argmax(self.policy_net(self.state)).cpu().detach().numpy()
        if sample > eps_threshold:
            self.action = self.argmax
        else:
            self.action = np.random.randint(self.action_size)
        return self.action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:# or self.steps_done%self.batch_size != 0:
            return
        # print("Iteration", self.steps_done, "Updating parameters.")
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        next_state_batch = torch.stack(batch.next_state) 
        reward_batch = torch.stack(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1).values
        # Compute the expected Q values
        next_state_values = next_state_values.unsqueeze(1)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        reward = np.array([outcomes[-1] * (values[-1] - prices[-1])], dtype=np.float32)
        if first_prices[-1] == second_prices[-1]:
            reward = np.array([(values[-1] - prices[-1]) / 2], dtype=np.float32)
        # if outcomes[-1] == 0:
        #     # Underbid regret
        #     reward = min(- (self.value - prices[-1]),0)
        #     # regret = - prices[-1]
        #     # print("regret", regret)
        # else:
        #     # Overbid regret
        #     reward = min(-(prices[-1] - second_prices[-1]),0)
        others_bid = (1 - outcomes[-1]) * first_prices[-1] + outcomes[-1] * second_prices[-1]
        # reward = np.array([(values[-1] - prices[-1]) * np.exp((bids[-1]) / 10)/ (np.exp((bids[-1]) / 2) + np.exp((others_bid / 2)))], dtype=np.float32)
        # reward = np.array([(values[-1] - prices[-1]) * ((bids[-1]))/ (((bids[-1])) + ((others_bid)))], dtype=np.float32)

        next_state =  np.array([bids[-1], others_bid], dtype=np.float32) 

        # converted_prices = self.convert_prices(next_state)
        # d = self.demand(converted_prices)
        # pi = (converted_prices - 1) * d
        # reward = pi[0]
        # print(reward)#, pi[0])

        next_state = torch.tensor(self.state_features(next_state)).to(device)
        
        reward = torch.tensor(reward, device=device)
 
        # Store the transition in memory
        self.memory.push(self.state, torch.tensor(self.action, device=device).reshape(1), next_state, reward)

        # Move to the next state
        self.state = next_state

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

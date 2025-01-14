import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Impression import ImpressionOpportunity
from Models import BidShadingContextualBandit, BidShadingPolicy, PyTorchWinRateEstimator


class Bidder:
    """ Bidder base class"""
    def __init__(self, rng):
        self.rng = rng
        self.truthful = False # Default

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        pass

    def clear_logs(self, memory):
        pass

class NoisyBidder(Bidder):
    """ A bidder that uses Q-learning to bid """
    def __init__(self, rng, value):
        super(NoisyBidder, self).__init__(rng)
        self.truthful = True
        # assume value is 100 for the naive example
        # the bid is integer
        self.value = value
        self.action = np.random.randint(value)
        self.epsilon = 0.3
        self.decay = 0.00002
        



    def bid(self, value, context, estimated_CTR=1):
        assert(estimated_CTR == 1)
        
        # print(coin_flip, self.epsilon, self.action)
        action = int(np.random.uniform(low=0, high=self.value + 1, size=1))#) int(np.random.normal(self.value-2, 1, 1))
        self.argmax = action
        return action

    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # print("outcomesoutcomes", outcomes)
        # coin_flip = np.random.rand(1)
        # if coin_flip >= self.epsilon:
        #     # exploit
        #     return
        # # explore
        # reward = outcomes[-1] * (values[-1] - prices[-1])
        # if reward == 0:
        #     self.action += 1
        # else:
        #     self.action = int(1 * self.action / 2)
        return
        

class HeuristicBidder(Bidder):
    """ A bidder that uses Q-learning to bid """
    def __init__(self, rng, value):
        super(HeuristicBidder, self).__init__(rng)
        self.truthful = True
        # assume value is 100 for the naive example
        # the bid is integer
        self.value = value
        self.action = 0 # np.random.randint(value)
        self.epsilon = 0.3
        self.decay = 0.00002
        



    def bid(self, value, context, estimated_CTR=1):
        assert(estimated_CTR == 1)
        
        # print(coin_flip, self.epsilon, self.action)
        self.argmax = self.action
        return self.action


    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # print("outcomesoutcomes", outcomes)
        # coin_flip = np.random.rand(1)
        # if coin_flip >= self.epsilon:
        #     # exploit
        #     return
        # # explore
        # reward = outcomes[-1] * (values[-1] - prices[-1])
        # if reward == 0:
        #     self.action += 1
        # else:
        #     self.action = int(1 * self.action / 2)
        self.action = self.value - self.action #self.value/2 - self.action


class ApproximateQMixedRegretBidder(Bidder):
    def __init__(self, rng, value, epsilon, decay, alpha, gamma, reward=0.0):
        super(ApproximateQMixedRegretBidder, self).__init__(rng)
        self.truthful = True
        # Size of the state feature vector
        # including previous bid, previous first price, previous second price, and action
        self.state_size = 3
        self.value = value
        self.action_size = value + 1  # Number of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy parameter for exploration
        self.decay = decay # epsilon decay
        self.action = 10
        
        self.all_actions = np.expand_dims(np.arange(self.action_size), 1)
        self.n_basis = 3
        self.state = np.zeros(self.state_size)
        self.theta_reward = np.random.rand(self.action_size, (self.state_size)*self.n_basis)#, self.action_size)  # Weight parameters for the Q-function
        self.theta_regret = np.random.rand(self.action_size, (self.state_size)*self.n_basis)
        self.reward = reward

    def state_features(self, state):
        # fourier basis of n degrees
        state_actions = state/self.value - 0.5
        state_features = np.concatenate([np.cos(i * np.pi * state) for i in range(self.n_basis)], axis=-1)
        # print(state_features)
        return state_features

    def q_values(self, state, actions):
        # linear approximation
        return np.dot(self.state_features(state), self.theta_reward.T)[actions].squeeze(), np.dot(self.state_features(state), self.theta_regret.T)[actions].squeeze() # [20]

    def bid(self, value, context, estimated_CTR=1):
        coin_flip = np.random.rand(1)
        state_expanded = np.repeat(np.expand_dims(self.state,0), self.action_size, axis=0) # [20,3]
        # state_actions =  np.concatenate([state_expanded, self.all_actions], axis = 1) # [20, 4]
        q_values = self.q_values(self.state, self.all_actions)
        # print(q_values)


        q_values_chosen = q_values[1] if np.random.rand(1) > self.reward else q_values[0]

        self.argmax = np.argmax(q_values_chosen)# np.random.choice(np.where(q_values > np.max(q_values) - 0.01)[0])

        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
        else:
            # explore
            self.action = np.random.randint(self.action_size)
        self.q = q_values[0][self.action], q_values[1][self.action]
        # print(coin_flip, self.epsilon, self.action)
        return self.action

    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        reward = outcomes[-1] * (values[-1] - prices[-1])
        # print("reward", reward)
        if outcomes[-1] == 0:
            # Underbid regret
            # regret = 0#- (self.value - prices[-1])
            regret = min(- (self.value - first_prices[-1]), 0)
        else:
            # Overbid regret
            regret = min(-(prices[-1] - second_prices[-1]), 0)
        regret = self.reward * reward + (1 - self.reward) * regret
        next_state = np.array([bids[-1], first_prices[-1], second_prices[-1]])#np.array([bids[-1], prices[-1]]) #
        # next_state_expanded = np.repeat(np.expand_dims(next_state,0), self.action_size, axis=0) # [20,3]
        # next_features =  np.concatenate([next_state_expanded, self.all_actions], axis = 1) # [20, 4]
        next_q_values = self.q_values(next_state, self.all_actions) # next_q_values = np.dot(next_features, np.expand_dims(self.theta, 1)).squeeze() # [20]
        target = regret + self.gamma * np.max(next_q_values[0]), regret + self.gamma * np.max(next_q_values[1])
        td = target[0] - self.q[0], target[1] - self.q[1]
        state_features = self.state_features(self.state) #np.concatenate([], axis=-1)
        self.theta_reward[bids[-1]] += self.alpha * td[0] * state_features
        self.theta_regret[bids[-1]] += self.alpha * td[1] * state_features
        self.state = next_state
        self.epsilon *= np.exp(-self.decay) # exploit more and more
 

class ApproximateQInterpolateBidder(Bidder):
    def __init__(self, rng, value, epsilon, decay, alpha, gamma, reward=0.0):
        super(ApproximateQInterpolateBidder, self).__init__(rng)
        self.truthful = True
        # Size of the state feature vector
        # including previous bid, previous first price, previous second price, and action
        self.state_size = 3
        self.value = value
        self.action_size = value + 1  # Number of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy parameter for exploration
        self.decay = decay # epsilon decay
        self.action = 10
        
        self.all_actions = np.expand_dims(np.arange(self.action_size), 1)
        self.n_basis = 3
        self.state = np.zeros(self.state_size)
        self.theta = np.random.rand(self.action_size, (self.state_size)*self.n_basis)#, self.action_size)  # Weight parameters for the Q-function
        self.reward = reward

    def state_features(self, state):
        # fourier basis of n degrees
        state_actions = state/self.value - 0.5
        state_features = np.concatenate([np.cos(i * np.pi * state) for i in range(self.n_basis)], axis=-1)
        # print(state_features)
        return state_features

    def q_values(self, state, actions):
        # linear approximation
        return np.dot(self.state_features(state), self.theta.T)[actions].squeeze() # [20]

    def bid(self, value, context, estimated_CTR=1):
        coin_flip = np.random.rand(1)
        state_expanded = np.repeat(np.expand_dims(self.state,0), self.action_size, axis=0) # [20,3]
        # state_actions =  np.concatenate([state_expanded, self.all_actions], axis = 1) # [20, 4]
        q_values = self.q_values(self.state, self.all_actions)
        # print(q_values)
        self.argmax = np.argmax(q_values)# np.random.choice(np.where(q_values > np.max(q_values) - 0.01)[0])

        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
        else:
            # explore
            self.action = np.random.randint(self.action_size)
        self.q = q_values[self.action]
        # print(coin_flip, self.epsilon, self.action)
        return self.action

    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        reward = outcomes[-1] * (values[-1] - prices[-1])
        # print("reward", reward)
        if outcomes[-1] == 0:
            # Underbid regret
            # regret = 0#- (self.value - prices[-1])
            regret = min(- (self.value - first_prices[-1]), 0)
        else:
            # Overbid regret
            regret = min(-(prices[-1] - second_prices[-1]), 0)
        regret = self.reward * reward + (1 - self.reward) * regret
        next_state = np.array([bids[-1], first_prices[-1], second_prices[-1]])#np.array([bids[-1], prices[-1]]) #
        # next_state_expanded = np.repeat(np.expand_dims(next_state,0), self.action_size, axis=0) # [20,3]
        # next_features =  np.concatenate([next_state_expanded, self.all_actions], axis = 1) # [20, 4]
        next_q_values = self.q_values(next_state, self.all_actions) # next_q_values = np.dot(next_features, np.expand_dims(self.theta, 1)).squeeze() # [20]
        target = regret + self.gamma * np.max(next_q_values)
        td = target - self.q
        state_features = self.state_features(self.state) #np.concatenate([], axis=-1)
        self.theta[bids[-1]] += self.alpha * td * state_features
        self.state = next_state
        self.epsilon *= np.exp(-self.decay) # exploit more and more
 

class ApproximateRegretQBidder(Bidder):
    def __init__(self, rng, value, epsilon, decay, alpha, gamma):
        super(ApproximateRegretQBidder, self).__init__(rng)
        self.truthful = True
        # Size of the state feature vector
        # including previous bid, previous first price, previous second price, and action
        self.state_size = 3
        self.value = value
        self.action_size = value + 1  # Number of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy parameter for exploration
        self.decay = decay # epsilon decay
        self.action = 10
        
        self.all_actions = np.expand_dims(np.arange(self.action_size), 1)
        self.n_basis = 3
        self.state = np.zeros(self.state_size)
        self.theta = np.random.rand(self.action_size, (self.state_size)*self.n_basis)#, self.action_size)  # Weight parameters for the Q-function


    def state_features(self, state):
        # fourier basis of n degrees
        state_actions = state/self.value - 0.5
        state_features = np.concatenate([np.cos(i * np.pi * state) for i in range(self.n_basis)], axis=-1)
        # print(state_features)
        return state_features

    def q_values(self, state, actions):
        # linear approximation
        return np.dot(self.state_features(state), self.theta.T)[actions].squeeze() # [20]

    def bid(self, value, context, estimated_CTR=1):
        coin_flip = np.random.rand(1)
        state_expanded = np.repeat(np.expand_dims(self.state,0), self.action_size, axis=0) # [20,3]
        # state_actions =  np.concatenate([state_expanded, self.all_actions], axis = 1) # [20, 4]
        q_values = self.q_values(self.state, self.all_actions)
        # print(q_values)
        self.argmax = np.argmax(q_values)# np.random.choice(np.where(q_values > np.max(q_values) - 0.01)[0])

        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
        else:
            # explore
            self.action = np.random.randint(self.action_size)
        self.q = q_values[self.action]
        # print(coin_flip, self.epsilon, self.action)
        return self.action

    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        reward = outcomes[-1] * (values[-1] - prices[-1])
        # print("reward", reward)
        if outcomes[-1] == 0:
            # Underbid regret
            # regret = 0#- (self.value - prices[-1])
            regret = min(- (self.value - first_prices[-1]), 0)
        else:
            # Overbid regret
            regret = min(-(prices[-1] - second_prices[-1]), 0)
        # regret = 0.25 * reward + 0.75 * regret
        next_state = np.array([bids[-1], first_prices[-1], second_prices[-1]])#np.array([bids[-1], prices[-1]]) #
        # next_state_expanded = np.repeat(np.expand_dims(next_state,0), self.action_size, axis=0) # [20,3]
        # next_features =  np.concatenate([next_state_expanded, self.all_actions], axis = 1) # [20, 4]
        next_q_values = self.q_values(next_state, self.all_actions) # next_q_values = np.dot(next_features, np.expand_dims(self.theta, 1)).squeeze() # [20]
        target = regret + self.gamma * np.max(next_q_values)
        td = target - self.q
        state_features = self.state_features(self.state) #np.concatenate([], axis=-1)
        self.theta[bids[-1]] += self.alpha * td * state_features
        self.state = next_state
        self.epsilon *= np.exp(-self.decay) # exploit more and more
 

class ApproximateQBidder(Bidder):
    def __init__(self, rng, value, epsilon, decay, alpha, gamma, deviate=False):
        super(ApproximateQBidder, self).__init__(rng)
        self.truthful = True
        # Size of the state feature vector
        # including previous bid, previous first price, previous second price, and action
        self.state_size = 3
        self.value = value
        self.action_size = value + 1  # Number of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.original_epsilon = epsilon
        self.epsilon = epsilon  # Epsilon-greedy parameter for exploration
        self.decay = decay # epsilon decay
        self.action = 10
        
        self.all_actions = np.expand_dims(np.arange(self.action_size), 1)
        self.n_basis = 3
        self.state = np.zeros(self.state_size)
        self.theta = np.random.rand(self.action_size, (self.state_size)*self.n_basis)#, self.action_size)  # Weight parameters for the Q-function
        self.previous_action = 0
        self.previous_previous_action = 0
        self.deviate = deviate
        self.same_bid = 0

    def state_features(self, state):
        # fourier basis of n degrees
        state_actions = state/self.action_size - 0.5
        state_features = np.concatenate([np.cos(i * np.pi * state) for i in range(self.n_basis)], axis=-1)
        # print(state_features)
        return state_features

    def q_values(self, state, actions):
        # linear approximation
        return np.dot(self.state_features(state), self.theta.T)[actions].squeeze() # [20]

    def bid(self, value, context, estimated_CTR=1):
        coin_flip = np.random.rand(1)
        state_expanded = np.repeat(np.expand_dims(self.state,0), self.action_size, axis=0) # [20,3]
        # state_actions =  np.concatenate([state_expanded, self.all_actions], axis = 1) # [20, 4]
        q_values = self.q_values(self.state, self.all_actions)
        # print(q_values)
        self.argmax =  np.random.choice(np.where(q_values > np.max(q_values) - 0.01)[0])
        # np.argmax(q_values)#
        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
        else:
            # explore
            self.action = np.random.randint(self.action_size)
        self.q = q_values[self.action]
        # print(coin_flip, self.epsilon, self.action)
        return self.action

    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        reward = outcomes[-1] * (values[-1] - prices[-1])
        # print("reward", reward)
        # if outcomes[-1] == 0:
        #     # Underbid regret
        #     # regret = 0#- (self.value - prices[-1])
        #     regret = - prices[-1]#- (self.value )
        #     # print("regret", regret)
        # else:
        #     # Overbid regret
        #     regret = -(prices[-1] - second_prices[-1])
        next_state = np.array([bids[-1], first_prices[-1], second_prices[-1]])
        # next_state = np.array([bids[-1], prices[-1]])
        # next_state = np.array([bids[-1]])
        # next_state_expanded = np.repeat(np.expand_dims(next_state,0), self.action_size, axis=0) # [20,3]
        # next_features =  np.concatenate([next_state_expanded, self.all_actions], axis = 1) # [20, 4]
        next_q_values = self.q_values(next_state, self.all_actions) # next_q_values = np.dot(next_features, np.expand_dims(self.theta, 1)).squeeze() # [20]
        target = reward + self.gamma * np.max(next_q_values)
        td = target - self.q
        state_features = self.state_features(self.state) #np.concatenate([], axis=-1)
        self.theta[bids[-1]] += self.alpha * td * state_features
        self.state = next_state
        if self.deviate:#(self.bid == self.previous_action or self.bid == self.previous_previous_action):
            self.same_bid += 1
        # else:
        #     self.same_bid = 0
        if self.same_bid > 275000:
            self.epsilon = self.original_epsilon
            self.same_bid = 0
        self.previous_previous_action = self.previous_action
        self.previous_action = self.bid
        self.epsilon *= np.exp(-self.decay) # exploit more and more
        

class VectorQRegretBidder(Bidder):
    """ A bidder that uses Q-learning to bid """
    def __init__(self, rng, value, epsilon, decay, alpha, gamma):
        super(VectorQRegretBidder, self).__init__(rng)
        self.truthful = True
        # assume value is 100 for the naive example
        # the bid is integer
        self.value = value
        self.q_table = torch.zeros((self.value), dtype=float)#torch.randint(0, self.value, size=(self.value, self.value), dtype=float)
        self.epsilon = epsilon
        self.decay = decay # 0.000002 #0.9999999
        self.alpha = alpha #0.001
        self.gamma = gamma #0.9999
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.rounds = 0


    def bid(self, value, context, estimated_CTR=1):
        assert(estimated_CTR == 1)
        coin_flip = np.random.rand(1)
        self.argmax = np.random.choice(torch.where(self.q_table == torch.max(self.q_table))[0])
        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
            self.next_state = self.action
        else:
            # explore
            self.action = np.random.randint(self.value)
            self.next_state = self.action
        # print(coin_flip, self.epsilon, self.action)
        return self.action


    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # print("outcomesoutcomes", outcomes)
        if outcomes[-1] == 0:
            # Underbid regret
            regret = min(- (self.value - first_prices[-1]),0)
            # regret = - prices[-1]
            # print("regret", regret)
        else:
            # Overbid regret
            regret = min(-(prices[-1] - second_prices[-1]),0)
        self.q_table[self.action] += self.alpha * (regret + self.gamma * torch.max(self.q_table) - self.q_table[self.action])
        self.state = self.next_state
        # print(self.q_table[self.state, self.action])
        # print(reward, self.state, self.action)
        self.epsilon *= np.exp(-self.decay) # exploit more and more


class VectorQBidder(Bidder):
    """ A bidder that uses Q-learning to bid """
    def __init__(self, rng, value, epsilon, decay, alpha, gamma):
        super(VectorQBidder, self).__init__(rng)
        self.truthful = True
        # assume value is 100 for the naive example
        # the bid is integer
        self.value = value
        self.q_table = torch.zeros((self.value + 1), dtype=float)#torch.randint(0, self.value, size=(self.value, self.value), dtype=float)
        self.epsilon = epsilon
        self.decay = decay # 0.000002 #0.9999999
        self.alpha = alpha #0.001
        self.gamma = gamma #0.9999
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.rounds = 0
        



    def bid(self, value, context, estimated_CTR=1):
        assert(estimated_CTR == 1)
        coin_flip = np.random.rand(1)
        self.argmax = np.random.choice(torch.where(self.q_table == torch.max(self.q_table))[0])
        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
            self.next_state = self.action
        else:
            # explore
            self.action = np.random.randint(self.value)
            self.next_state = self.action
        # print(coin_flip, self.epsilon, self.action)
        return self.action


    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # print("outcomesoutcomes", outcomes)
        reward = outcomes[-1] * (values[-1] - prices[-1])
        self.q_table[self.action] += self.alpha * (reward + self.gamma * torch.max(self.q_table) - self.q_table[self.action])
        self.state = self.next_state
        # print(self.q_table[self.state, self.action])
        # print(reward, self.state, self.action)
        self.epsilon *= np.exp(-self.decay) # exploit more and more


class TabularQObserveBothBidder(Bidder):
    def __init__(self, rng, value, epsilon, decay, alpha, gamma):
        super(TabularQObserveBothBidder, self).__init__(rng)
        self.truthful = True
        # Size of the state feature vector
        # including previous bid, previous first price, previous second price, and action
        self.value = value
        self.action_size = value  # Number of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon-greedy parameter for exploration
        self.decay = decay # epsilon decay
        self.action = 10
        self.q_table = np.zeros((self.value*self.value, self.value), dtype=float)#
        self.all_actions = np.expand_dims(np.arange(self.action_size), 1)
        self.state = np.zeros(2)


    def state_features(self, state):
        # fourier basis of n degrees
        state_features = int(state[0]* self.value + state[1])
        # print(state_features)
        return state_features

    def q_values(self, state, actions):
        # linear approximation
        return self.q_table[self.state_features(state), actions].flatten()

    def bid(self, value, context, estimated_CTR=1):
        coin_flip = np.random.rand(1)
        q_values = self.q_values(self.state, self.all_actions)
        # print(q_values)
        self.argmax = np.random.choice(np.where(q_values > np.max(q_values) - 0.01)[0])
        # np.argmax(q_values)#
        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
        else:
            # explore
            self.action = np.random.randint(self.action_size)
        self.q = q_values[self.action]
        # print(coin_flip, self.epsilon, self.action)
        return self.action

    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        reward = outcomes[-1] * (values[-1] - prices[-1])
        # if outcomes[-1] == 0:
        #     # Underbid regret
        #     reward = min(- (self.value - prices[-1]),0)
        #     # regret = - prices[-1]
        #     # print("regret", regret)
        # else:
        #     # Overbid regret
        #     reward = min(-(prices[-1] - second_prices[-1]),0)
        others_bid = (1 - outcomes[-1]) * first_prices[-1] + outcomes[-1] * second_prices[-1]
        next_state =  np.array([bids[-1], others_bid])#np.array([bids[-1], prices[-1]]) #
        next_q_values = self.q_values(next_state, self.all_actions) # next_q_values = np.dot(next_features, np.expand_dims(self.theta, 1)).squeeze() # [20]
        target = reward + self.gamma * np.max(next_q_values)
        td = target - self.q
        state_features = self.state_features(self.state) #np.concatenate([], axis=-1)
        self.q_table[state_features, bids[-1]] += self.alpha * td
        self.state = next_state
        self.epsilon *= np.exp(-self.decay) # exploit more and more
        


class TabularQBidder(Bidder):
    """ A bidder that uses Q-learning to bid """
    def __init__(self, rng, value, epsilon, decay, alpha, gamma):
        super(TabularQBidder, self).__init__(rng)
        self.truthful = True
        # assume value is 100 for the naive example
        # the bid is integer
        self.value = value
        self.q_table = torch.zeros((self.value, self.value), dtype=float)#torch.randint(0, self.value, size=(self.value, self.value), dtype=float)
        self.epsilon = epsilon
        self.decay = decay #0.9999999
        self.alpha = alpha
        self.gamma = gamma
        self.state = 0
        self.action = 0
        self.next_state = 0



    def bid(self, value, context, estimated_CTR=1):
        assert(estimated_CTR == 1)
        coin_flip = np.random.rand(1)
        self.argmax = np.random.choice(torch.where(self.q_table[self.state] == torch.max(self.q_table[self.state]))[0])
        if coin_flip >= self.epsilon:
            # exploit
            self.action = self.argmax
            self.next_state = self.action
        else:
            # explore
            self.action = np.random.randint(self.value)
            self.next_state = self.action
        # print(coin_flip, self.epsilon, self.action)
        return self.action


    def update(self, contexts, values, bids, prices, first_prices, second_prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # print("outcomesoutcomes", outcomes)
        # print(values[-1], bids[-1], prices[-1])
        reward = outcomes[-1] * (values[-1] - prices[-1])
        self.q_table[self.state,self.action] += self.alpha * (reward + self.gamma * torch.max(self.q_table[int(bids[-1])]) - self.q_table[self.state,self.action])
        self.state = int(bids[-1])
        # print(self.q_table[self.state, self.action])
        # print(reward, self.state, self.action)
        self.epsilon *= np.exp(-self.decay) # exploit more and more
        


class TruthfulBidder(Bidder):
    """ A bidder that bids truthfully """
    def __init__(self, rng):
        super(TruthfulBidder, self).__init__(rng)
        self.truthful = True

    def bid(self, value, context, estimated_CTR):
        return value * estimated_CTR


class EmpiricalShadedBidder(Bidder):
    """ A bidder that learns a single bidding factor gamma from past data """

    def __init__(self, rng, gamma_sigma, init_gamma=1.0):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.gammas = []
        super(EmpiricalShadedBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        # Sample the shade factor gamma
        gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
        if gamma < 0.0:
            gamma = 0.0
        if gamma > 1.0:
            gamma = 1.0
        bid *= gamma
        self.gammas.append(gamma)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]

        # Extract shading factors to numpy
        gammas = np.array(self.gammas)

        if plot:
            _,_=plt.subplots(figsize=figsize)
            plt.title('Raw observations',fontsize=fontsize+2)
            plt.scatter(gammas,utilities, alpha=.25)
            plt.xlabel(r'Shading factor ($\gamma$)',fontsize=fontsize)
            plt.ylabel('Net Utility',fontsize=fontsize)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)
            # plt.show()

        # We want to be able to estimate utility for any other continuous value, but this is a hassle in continuous space.
        # Instead -- we'll bucketise and look at the empirical utility distribution within every bucket
        min_gamma, max_gamma = np.min(gammas), np.max(gammas)
        grid_delta = .005
        num_buckets = int((max_gamma-min_gamma) // grid_delta) + 1
        buckets = np.linspace(min_gamma, max_gamma, num_buckets)
        x = []
        estimated_y_mean = []
        estimated_y_stderr = []
        bucket_lo = buckets[0]
        for idx, bucket_hi in enumerate(buckets[1:]):
            # Mean of the bucket
            x.append((bucket_hi-bucket_lo)/2.0 + bucket_lo)
            # Only look at samples within this range
            mask = np.logical_and(gammas < bucket_hi, bucket_lo <= gammas)
            # If we can draw meaningful inferences
            num_samples = len(utilities[mask])
            if num_samples > 1:
                # Extrapolate mean utility from these samples
                bucket_utility = utilities[mask].mean()
                estimated_y_mean.append(bucket_utility)
                # Compute standard error on utility estimate
                estimated_y_stderr.append(np.std(utilities[mask]) / np.sqrt(num_samples))
            else:
                estimated_y_mean.append(np.nan)
                estimated_y_stderr.append(np.nan)
            # Move sliding window for bucket
            bucket_lo = bucket_hi
        # To NumPy format
        x = np.asarray(x)
        estimated_y_mean = np.asarray(estimated_y_mean)
        estimated_y_stderr = np.asarray(estimated_y_stderr)

        # This is relatively high because we underestimate total variance
        # (1) Variance from click ~ Bernoulli(p)
        # (2) Variance from uncertainty about winning the auction
        critical_value = 1.96
        U_lower_bound = estimated_y_mean - critical_value * estimated_y_stderr

        # Move the mean of the policy towards the empirically best value
        # Search the array in reverse so we take the highest value in case of ties
        best_idx = len(x) - np.nanargmax(U_lower_bound[::-1]) - 1
        best_gamma = x[best_idx]
        if best_gamma < 0:
            best_gamma = 0
        if best_gamma > 1.0:
            best_gamma = 1.0
        self.prev_gamma = best_gamma

        if plot:
            fig, axes = plt.subplots(figsize=figsize)
            plt.suptitle(name, fontsize=fontsize+2)
            plt.title(f'Iteration: {iteration}', fontsize=fontsize)
            plt.plot(x, estimated_y_mean, label='Estimate', ls='--', color='red')
            plt.fill_between(x,
                             estimated_y_mean - critical_value * estimated_y_stderr,
                             estimated_y_mean + critical_value * estimated_y_stderr,
                             alpha=.25,
                             color='red',
                             label='C.I.')
            plt.axvline(best_gamma, ls='--', color='gray', label='Best')
            plt.axhline(0, ls='-.', color='gray')
            plt.xlabel(r'Multiplicative Bid Shading Factor ($\gamma$)', fontsize=fontsize)
            plt.ylabel('Estimated Net Utility', fontsize=fontsize)
            plt.ylim(-1.0, 2.0)
            plt.xticks(fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)
            plt.legend(fontsize=fontsize)
            plt.tight_layout()
            #plt.show()

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
        else:
            self.gammas = self.gammas[-memory:]


class ValueLearningBidder(Bidder):
    """ A bidder that estimates the optimal bid shading distribution via value learning """

    def __init__(self, rng, gamma_sigma, init_gamma=1.0, inference='search'):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        assert inference in ['search', 'policy']
        self.inference = inference
        self.gammas = []
        self.propensities = []
        self.winrate_model = PyTorchWinRateEstimator()
        self.bidding_policy = BidShadingPolicy() if inference == 'policy' else None
        self.model_initialised = False
        super(ValueLearningBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shadin factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.prev_gamma - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
        elif self.inference == 'search':
            # Option 2:
            # Predict P(win|gamma,value,P(click))
            # Use it to predict utility, maximise utility
            n_values_search = 128
            gamma_grid = self.rng.uniform(0.1, 1.0, size=n_values_search)
            gamma_grid.sort()
            x = torch.Tensor(np.hstack((np.tile(estimated_CTR, (n_values_search, 1)), np.tile(value, (n_values_search, 1)), gamma_grid.reshape(-1,1))))

            prob_win = self.winrate_model(x).detach().numpy().ravel()

            # U = W (V - P)
            expected_value = bid
            shaded_bids = expected_value * gamma_grid
            estimated_utility = prob_win * (expected_value - shaded_bids)
            gamma = gamma_grid[np.argmax(estimated_utility)]
            propensity = 1.0

        elif self.inference == 'policy':
            # Option 3: sample from the learnt policy instead of brute force searching
            x = torch.Tensor([estimated_CTR, value])
            with torch.no_grad():
                gamma, propensity = self.bidding_policy(x)
                gamma = gamma.detach().item()

        bid *= gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # FALLBACK: if you lost every auction you participated in, your model collapsed
        # Revert to not shading for 1 round, to collect data with informational value
        if not won_mask.astype(np.uint8).sum():
            self.model_initialised = False
            print(f'! Fallback for {name}')
            return

        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        utilities = torch.Tensor(utilities)

        # Augment data with samples: if you shade 100%, you will lose
        # If you won now, you would have also won if you bid higher
        X = np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), np.array(self.gammas).reshape(-1, 1)))

        X_aug_neg = X.copy()
        X_aug_neg[:, -1] = 0.0

        X_aug_pos = X[won_mask].copy()
        X_aug_pos[:, -1] = np.maximum(X_aug_pos[:, -1], 1.0)

        X = torch.Tensor(np.vstack((X, X_aug_neg)))

        y = won_mask.astype(np.uint8).reshape(-1,1)
        y = torch.Tensor(np.concatenate((y, np.zeros_like(y))))

        # Fit the model
        self.winrate_model.train()
        epochs = 8192 * 4
        lr = 3e-3
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-7, factor=0.1, verbose=True)
        criterion = torch.nn.BCELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()
            pred_y = self.winrate_model(X)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 512:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)

        self.winrate_model.eval()
        fig, ax = plt.subplots()
        plt.title(f'{name}')
        plt.plot(losses, label=r'P(win|$gamma$,x)')
        plt.ylabel('Loss')
        plt.legend()
        fig.set_tight_layout(True)
        # plt.show()

        # Predict Utility -- \hat{u}
        orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), np.array(self.gammas).reshape(-1, 1))))
        W = self.winrate_model(orig_features).squeeze().detach().numpy()
        print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

        if self.inference == 'policy':
            # Learn a policy to maximise E[U | bid] where bid ~ policy
            X = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1))))

            self.bidding_policy.train()
            epochs = 8192 * 2
            lr = 2e-3
            optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-7, factor=0.1, verbose=True)
            losses = []
            best_epoch, best_loss = -1, np.inf
            for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
                optimizer.zero_grad()
                # Sample bid shading values
                sampled_gamma, propensities = self.bidding_policy(X)

                # Add them to input for win probability model
                X_with_gamma = torch.hstack((X, sampled_gamma))

                # Estimate utility for these sampled bid shading values
                prob_win = self.winrate_model(X_with_gamma).squeeze()
                values = X_with_gamma[:, 0].squeeze() * X_with_gamma[:, 1].squeeze()
                prices = values * sampled_gamma.squeeze()

                estimated_utility = -(prob_win * (values - prices)).mean()
                estimated_utility.backward()
                optimizer.step()

                losses.append(estimated_utility.item())
                scheduler.step(estimated_utility)
                if (best_loss - losses[-1]) > 1e-6:
                    best_epoch = epoch
                    best_loss = losses[-1]
                elif epoch - best_epoch > 256:
                    print(f'Stopping at Epoch {epoch}')
                    break

            losses = np.array(losses)
            self.bidding_policy.eval()
            fig, ax = plt.subplots()
            plt.title(f'{name}')
            plt.plot(losses, label=r'$\pi(\gamma)$')
            plt.ylabel('- Estimated Expected Utility')
            plt.legend()
            fig.set_tight_layout(True)
            #plt.show()

        self.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]


class PolicyLearningBidder(Bidder):
    """ A bidder that estimates the optimal bid shading distribution via policy learning """

    def __init__(self, rng, gamma_sigma, loss, init_gamma=1.0):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.gammas = []
        self.propensities = []
        self.model = BidShadingContextualBandit(loss)
        self.model_initialised = False
        super(PolicyLearningBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shading factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.prev_gamma - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
        else:
            # Option 2:
            # Sample from the contextual bandit
            x = torch.Tensor([estimated_CTR, value])
            gamma, propensity = self.model(x)
            gamma = torch.clip(gamma, 0.0, 1.0)

        bid *= gamma.detach().item() if self.model_initialised else gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return int(bid)

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        utilities = torch.Tensor(utilities)

        # Extract shading factors to torch
        gammas = torch.Tensor(self.gammas)

        # Prepare features
        X = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1))))

        if not self.model_initialised:
            self.model.initialise_policy(X, gammas)

        # Ensure we don't have propensities that are rounded to zero
        propensities = torch.clip(torch.Tensor(self.propensities), min=1e-15)

        # Fit the model
        self.model.train()
        epochs = 8192 * 2
        lr = 2e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, verbose=True)

        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()  # Setting our stored gradients equal to zero
            loss = self.model.loss(X, gammas, propensities, utilities, importance_weight_clipping_eps=50.0)
            loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
            optimizer.step()  # Updates weights and biases with the optimizer (SGD)
            losses.append(loss.item())
            scheduler.step(loss)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 512:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)
        if np.isnan(losses).any():
            print('NAN DETECTED! in losses')
            print(list(losses))
            print(np.isnan(X.detach().numpy()).any(), X)
            print(np.isnan(gammas.detach().numpy()).any(), gammas)
            print(np.isnan(propensities.detach().numpy()).any(), propensities)
            print(np.isnan(utilities.detach().numpy()).any(), utilities)
            exit(1)

        self.model.eval()
        expected_utility = -self.model.loss(X, gammas, propensities, utilities, KL_weight=0.0).detach().numpy()
        print('Expected utility:', expected_utility)

        pred_gammas, _ = self.model(X)
        pred_gammas = pred_gammas.detach().numpy()
        print(name, 'Number of samples: ', X.shape)
        print(name, 'Predicted Gammas: ', pred_gammas.min(), pred_gammas.max(), pred_gammas.mean())

        self.model_initialised = True
        self.model.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]


class DoublyRobustBidder(Bidder):
    """ A bidder that estimates the optimal bid shading distribution with a Doubly Robust Estimator """

    def __init__(self, rng, gamma_sigma, init_gamma=1.0):
        self.gamma_sigma = gamma_sigma
        self.prev_gamma = init_gamma
        self.gammas = []
        self.propensities = []
        self.winrate_model = PyTorchWinRateEstimator()
        self.bidding_policy = BidShadingContextualBandit(loss='Doubly Robust', winrate_model=self.winrate_model)
        self.model_initialised = False
        super(DoublyRobustBidder, self).__init__(rng)

    def bid(self, value, context, estimated_CTR):
        # Compute the bid as expected value
        bid = value * estimated_CTR
        if not self.model_initialised:
            # Option 1:
            # Sample the bid shading factor 'gamma' from a Gaussian
            gamma = self.rng.normal(self.prev_gamma, self.gamma_sigma)
            normal_pdf = lambda g: np.exp(-((self.prev_gamma - g) / self.gamma_sigma)**2/2) / (self.gamma_sigma * np.sqrt(2 * np.pi))
            propensity = normal_pdf(gamma)
        else:
            # Option 2:
            # Sample from the contextual bandit
            x = torch.Tensor([estimated_CTR, value])
            with torch.no_grad():
                gamma, propensity = self.bidding_policy(x)
                gamma = torch.clip(gamma, 0.0, 1.0)

        bid *= gamma.detach().item() if self.model_initialised else gamma
        self.gammas.append(gamma)
        self.propensities.append(propensity)
        return bid

    def update(self, contexts, values, bids, prices, outcomes, estimated_CTRs, won_mask, iteration, plot, figsize, fontsize, name):
        # Compute net utility
        utilities = np.zeros_like(values)
        utilities[won_mask] = (values[won_mask] * outcomes[won_mask]) - prices[won_mask]
        # utilities = torch.Tensor(utilities)

        ##############################
        # 1. TRAIN UTILITY ESTIMATOR #
        ##############################
        gammas_numpy = np.array([g.detach().item() if self.model_initialised else g for g in self.gammas])
        if self.model_initialised:
            # Predict Utility -- \hat{u}
            orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1))))
            W = self.winrate_model(orig_features).squeeze().detach().numpy()
            print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

            V = estimated_CTRs * values
            P = estimated_CTRs * values * gammas_numpy
            estimated_utilities = W * (V - P)

            errors = estimated_utilities - utilities
            print('Estimated Utility\t Mean Error:\t\t\t', errors.mean())
            print('Estimated Utility\t Mean Absolute Error:\t', np.abs(errors).mean())

        # Augment data with samples: if you shade 100%, you will lose
        # If you won now, you would have also won if you bid higher
        X = np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1)))

        X_aug_neg = X.copy()
        X_aug_neg[:, -1] = 0.0

        X_aug_pos = X[won_mask].copy()
        X_aug_pos[:, -1] = np.maximum(X_aug_pos[:, -1], 1.0)

        X = torch.Tensor(np.vstack((X, X_aug_neg)))

        y = won_mask.astype(np.uint8).reshape(-1,1)
        y = torch.Tensor(np.concatenate((y, np.zeros_like(y))))

        # Fit the model
        self.winrate_model.train()
        epochs = 8192 * 4
        lr = 3e-3
        optimizer = torch.optim.Adam(self.winrate_model.parameters(), lr=lr, weight_decay=1e-6, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=256, min_lr=1e-7, factor=0.2, verbose=True)
        criterion = torch.nn.BCELoss()
        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()
            pred_y = self.winrate_model(X)
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            scheduler.step(loss)
            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 1024:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)

        self.winrate_model.eval()

        # Predict Utility -- \hat{u}
        orig_features = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1), gammas_numpy.reshape(-1, 1))))
        W = self.winrate_model(orig_features).squeeze().detach().numpy()
        print('AUC predicting P(win):\t\t\t\t', roc_auc_score(won_mask.astype(np.uint8), W))

        V = estimated_CTRs * values
        P = estimated_CTRs * values * gammas_numpy
        estimated_utilities = W * (V - P)

        errors = estimated_utilities - utilities
        print('Estimated Utility\t Mean Error:\t\t\t', errors.mean())
        print('Estimated Utility\t Mean Absolute Error:\t', np.abs(errors).mean())

        ##############################
        # 2. TRAIN DOUBLY ROBUST POLICY #
        ##############################
        utilities = torch.Tensor(utilities)
        estimated_utilities = torch.Tensor(estimated_utilities)
        gammas = torch.Tensor(self.gammas)

        # Prepare features
        X = torch.Tensor(np.hstack((estimated_CTRs.reshape(-1,1), values.reshape(-1,1))))

        if not self.model_initialised:
            self.bidding_policy.initialise_policy(X, gammas)

        # Ensure we don't have propensities that are rounded to zero
        propensities = torch.clip(torch.Tensor(self.propensities), min=1e-15)

        # Fit the model
        self.bidding_policy.train()
        epochs = 8192 * 4
        lr = 7e-3
        optimizer = torch.optim.Adam(self.bidding_policy.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, min_lr=1e-8, factor=0.2, threshold=5e-3, verbose=True)

        losses = []
        best_epoch, best_loss = -1, np.inf
        for epoch in tqdm(range(int(epochs)), desc=f'{name}'):
            optimizer.zero_grad()  # Setting our stored gradients equal to zero
            loss = self.bidding_policy.loss(X, gammas, propensities, utilities, utility_estimates=estimated_utilities, winrate_model=self.winrate_model, importance_weight_clipping_eps=50.0)
            loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias
            optimizer.step()  # Updates weights and biases with the optimizer (SGD)
            losses.append(loss.item())
            scheduler.step(loss)

            if (best_loss - losses[-1]) > 1e-6:
                best_epoch = epoch
                best_loss = losses[-1]
            elif epoch - best_epoch > 512:
                print(f'Stopping at Epoch {epoch}')
                break

        losses = np.array(losses)
        if np.isnan(losses).any():
            print('NAN DETECTED! in losses')
            print(list(losses))
            print(np.isnan(X.detach().numpy()).any(), X)
            print(np.isnan(gammas.detach().numpy()).any(), gammas)
            print(np.isnan(propensities.detach().numpy()).any(), propensities)
            print(np.isnan(utilities.detach().numpy()).any(), utilities)
            exit(1)

        self.bidding_policy.eval()

        pred_gammas, _ = self.bidding_policy(X)
        pred_gammas = pred_gammas.detach().numpy()
        print(name, 'Number of samples: ', X.shape)
        print(name, 'Predicted Gammas: ', pred_gammas.min(), pred_gammas.max(), pred_gammas.mean())

        self.model_initialised = True
        self.bidding_policy.model_initialised = True

    def clear_logs(self, memory):
        if not memory:
            self.gammas = []
            self.propensities = []
        else:
            self.gammas = self.gammas[-memory:]
            self.propensities = self.propensities[-memory:]

from AuctionAllocation import AllocationMechanism
from Bidder import Bidder

import numpy as np

from BidderAllocation import OracleAllocator
from Models import sigmoid

class Auction:
    ''' Base class for auctions '''
    def __init__(self, rng, allocation, agents, agent2items, agents2item_values, max_slots, embedding_size, embedding_var, obs_embedding_size, num_participants_per_round):
        self.rng = rng
        self.allocation = allocation
        self.agents = agents
        self.max_slots = max_slots
        self.revenue = .0

        self.agent2items = agent2items
        self.agents2item_values = agents2item_values

        self.embedding_size = embedding_size
        self.embedding_var = embedding_var

        self.obs_embedding_size = obs_embedding_size

        self.num_participants_per_round = num_participants_per_round
        self.win_counts = {}

    def simulate_opportunity(self):
        # Sample the number of slots uniformly between [1, max_slots]
        num_slots = self.rng.integers(1, self.max_slots + 1)

        # Sample a true context vector
        true_context = np.concatenate((self.rng.normal(0, self.embedding_var, size=self.embedding_size), [1.0]))

        # Mask true context into observable context
        obs_context = np.concatenate((true_context[:self.obs_embedding_size], [1.0]))

        # At this point, the auctioneer solicits bids from
        # the list of bidders that might want to compete.
        bids = []
        CTRs = []
        participating_agents_idx = self.rng.choice(len(self.agents), self.num_participants_per_round, replace=False)
        participating_agents = [self.agents[idx] for idx in participating_agents_idx]
        for agent in participating_agents:
            # Get the bid and the allocated item
            if isinstance(agent.allocator, OracleAllocator):
                bid, item = agent.bid(true_context)
            else:
                bid, item = agent.bid(obs_context)
            bids.append(bid)
            # Compute the true CTRs for items in this agent's catalogue
            true_CTR = np.ones(len(self.agent2items[agent.name]))# sigmoid(true_context @ self.agent2items[agent.name].T)
            agent.logs[-1].set_true_CTR(np.max(true_CTR * self.agents2item_values[agent.name]), true_CTR[item])
            CTRs.append(true_CTR[item])
        bids = np.array(bids)
        CTRs = np.array(CTRs)
        # Now we have bids, we need to somehow allocate slots
        # "second_prices" tell us how much lower the winner could have gone without changing the outcome
        winners, prices, first_prices, second_prices = self.allocation.allocate(bids, num_slots, np.array([self.win_counts.get(key,0) for key in participating_agents]))
        # print(winners)
        for winner in winners:
            # win_counts_np = np.array(list(self.win_counts.values()))
            # if sum(win_counts_np) < 10:
            for key in self.win_counts.keys():
                self.win_counts[key] = 0.99 * self.win_counts[key]
            self.win_counts[participating_agents[winner]] = self.win_counts.get(participating_agents[winner],0) + 1

            # elif np.sum(win_counts_np) - self.win_counts.get(participating_agents[winner],0) > 0:
            #     self.win_counts[participating_agents[winner]] = self.win_counts.get(participating_agents[winner],0) + 2
            #     for key in self.win_counts.keys():
            #         self.win_counts[key] -= 1
        if len(second_prices) == 0:
            second_prices = prices
        # print("first_prices", first_prices)
        # Bidders only obtain value when they get their outcome
        # Either P(view), P(click | view, ad), P(conversion | click, view, ad)
        # For now, look at P(click | ad) * P(view)
        outcomes = self.rng.binomial(1, CTRs[winners])
        # print("winners", winners)
        # Let bidders know what they're being charged for
        for slot_id, (winner, price, first_price, second_price, outcome) in enumerate(zip(winners, prices, first_prices, second_prices, outcomes)):
            for agent_id, agent in enumerate(participating_agents):
                if agent_id == winner:
                    agent.charge(price, first_price, second_price, bool(outcome))
                else:
                    agent.set_price(price, first_price, second_price)
            self.revenue += price

    def clear_revenue(self):
        self.revenue = 0.0

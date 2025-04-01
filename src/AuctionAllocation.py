import numpy as np

class AllocationMechanism:
    ''' Base class for allocation mechanisms '''
    def __init__(self):
        pass

    def allocate(self, bids, num_slots):
        pass


class FirstPrice(AllocationMechanism):
    ''' (Generalised) First-Price Allocation '''

    def __init__(self):
        super(FirstPrice, self).__init__()

    def allocate(self, bids, num_slots, win_counts):
        winners = np.argsort(-bids)[:num_slots]
        sorted_bids = -np.sort(-bids)
        prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1:num_slots+1]
        return winners, prices, prices, second_prices


class SecondPrice(AllocationMechanism):
    ''' (Generalised) Second-Price Allocation '''

    def __init__(self):
        super(SecondPrice, self).__init__()

    def allocate(self, bids, num_slots, win_counts):
        winners = np.argsort(-bids)[:num_slots]
        sorted_bids = -np.sort(-bids)
        first_prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1:num_slots+1]
        return winners, second_prices, first_prices, second_prices


class MixedPrice(AllocationMechanism):
    ''' (Generalised) Second-Price Allocation '''

    def __init__(self):
        super(MixedPrice, self).__init__()
        self.mix_rate = 0.0

    def allocate(self, bids, num_slots, win_counts):
        cumulative_advantage = win_counts#win_counts #np.exp(win_counts - np.average(win_counts))
        avg_bid = np.average(bids)
        p_bid = np.exp(bids-avg_bid)/np.sum(np.exp(bids-avg_bid))  #(bids + 1) / np.sum(bids + 1)
        p_cum = (cumulative_advantage + 1)/(np.sum(cumulative_advantage + 1))
        # print(cumulative_advantage)
        p=(p_bid * p_cum)/np.sum(p_bid * p_cum)
        winners = np.random.choice(len(bids), num_slots, p=p)
        
        # winners = np.argsort(-bids)[:num_slots]
        print(p)
        print(winners)
        print("win_counts", win_counts)
        sorted_bids = -np.sort(-bids)
        first_prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1:num_slots+1]
        price = bids[winners]#((1-self.mix_rate)*first_prices + self.mix_rate*second_prices)
        return winners, price, first_prices, second_prices

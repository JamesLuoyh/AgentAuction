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

    def allocate(self, bids, num_slots):
        winners = np.argsort(-bids)[:num_slots]
        sorted_bids = -np.sort(-bids)
        prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1:num_slots+1]
        return winners, prices, prices, second_prices


class SecondPrice(AllocationMechanism):
    ''' (Generalised) Second-Price Allocation '''

    def __init__(self):
        super(SecondPrice, self).__init__()

    def allocate(self, bids, num_slots):
        winners = np.argsort(-bids)[:num_slots]
        sorted_bids = -np.sort(-bids)
        first_prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1:num_slots+1]
        return winners, second_prices, first_prices, second_prices


class MixedPrice(AllocationMechanism):
    ''' (Generalised) Second-Price Allocation '''

    def __init__(self):
        super(MixedPrice, self).__init__()
        self.fp_rate = 0.0

    def allocate(self, bids, num_slots):
        winners = np.argsort(-bids)[:num_slots]
        sorted_bids = -np.sort(-bids)
        first_prices = sorted_bids[:num_slots]
        second_prices = sorted_bids[1:num_slots+1]
        price = (self.fp_rate*first_prices + (1-self.fp_rate)*second_prices)
        return winners, price, first_prices, second_prices

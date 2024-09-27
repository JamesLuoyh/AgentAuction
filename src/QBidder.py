import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from Impression import ImpressionOpportunity

import Bidder

class QBidder(Bidder):
    """ A bidder that uses Q-learning to bid """
    def __init__(self, rng):
        super(QBidder, self).__init__(rng)
        self.truthful = False

    def bid(self, value, context, estimated_CTR=1):
        assert(estimated_CTR == 1)
        
        return value * estimated_CTR



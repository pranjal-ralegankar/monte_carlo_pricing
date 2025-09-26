

import numpy as np
from utils import bs_price

def european_payoff(paths, K, call=True):
    S_T = paths[:, -1]
    return np.maximum(S_T - K, 0) if call else np.maximum(K - S_T, 0)

def asian_arithmetic_payoff(paths, K, call=True):
    avg = paths.mean(axis=1)
    return np.maximum(avg - K, 0) if call else np.maximum(K - avg, 0)

def asian_geometric_payoff(paths, K, call=True):
    geo = np.exp(np.log(paths).mean(axis=1))
    return np.maximum(geo - K, 0) if call else np.maximum(K - geo, 0)

def barrier_up_and_out_payoff(paths, K, barrier, call=True, rebate=0.0):
    knocked = paths.max(axis=1) >= barrier
    payoff = european_payoff(paths, K, call)
    payoff[knocked] = rebate
    return payoff

def lookback_floating_payoff(paths, call=True):
    S_T = paths[:, -1]
    S_min = paths.min(axis=1)
    S_max = paths.max(axis=1)
    return np.maximum(S_T - S_min, 0) if call else np.maximum(S_max - S_T, 0)



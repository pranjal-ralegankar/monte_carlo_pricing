from math import sqrt
import numpy as np
from scipy.stats import norm

def bs_price(S0, K, r, sigma, T, call=True):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if call else K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def geometric_asian_closed_form(S0, K, r, sigma, T, n_steps, call=True):
    sigma_g = sigma * np.sqrt((n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2))
    return bs_price(S0, K, r, sigma_g, T, call)


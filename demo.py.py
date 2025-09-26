# File: demo.py
from models import simulate_gbm_paths
from payoffs import european_payoff
from utils import bs_price
from mc_engine import MonteCarloPricer

def demo():
    S0 = 100.0; K = 100.0; r = 0.01; sigma = 0.2; T = 1.0; n_steps = 200
    pricer = MonteCarloPricer(simulate_gbm_paths, discount_rate=r)

    # European option
    bs = bs_price(S0, K, r, sigma, T, call=True)
    res = pricer.price(european_payoff, S0, K, T, sigma, n_paths=50000, n_steps=n_steps, seed=1)
    print(f"European MC={res['price']:.4f} ± {res['stderr']*1.96:.4f}, BS={bs:.4f}")

if __name__ == '__main__':
    demo()
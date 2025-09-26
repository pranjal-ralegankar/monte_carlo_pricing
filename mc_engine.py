# File: mc_engine.py
import numpy as np

class MonteCarloPricer:
    def __init__(self, model_simulator, discount_rate=0.0):
        self.simulator = model_simulator
        self.r = discount_rate

    def price(self, payoff_fn, S0, K, T, sigma, r=None, n_paths=100000, n_steps=100, seed=None,
              antithetic=False, control_variate=None):
        r = r if r is not None else self.r
        rng = np.random.default_rng(seed)
        paths = self.simulator(S0, r, sigma, T, n_steps, n_paths, rng=rng, antithetic=antithetic)
        payoffs = payoff_fn(paths, K)
        disc_payoffs = np.exp(-r * T) * payoffs
        mean = disc_payoffs.mean()
        std_err = disc_payoffs.std(ddof=1) / np.sqrt(n_paths)
        result = {'price': mean, 'stderr': std_err}
        if control_variate is not None:
            cv_payoffs = control_variate['payoff'](paths, K)
            disc_cv = np.exp(-r * T) * cv_payoffs
            cov = np.cov(disc_payoffs, disc_cv)[0, 1]
            var_cv = disc_cv.var()
            b_star = cov / var_cv if var_cv != 0 else 0
            adjusted = disc_payoffs - b_star * (disc_cv - control_variate['true_mean'] * np.exp(-r * T))
            result.update({'cv_price': adjusted.mean(), 'cv_stderr': adjusted.std(ddof=1)/np.sqrt(n_paths)})
        return result

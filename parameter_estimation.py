from itertools import product

import numpy as np
from scipy.stats import norm, t

from robust_filter import Filter


def generate_param_grid(param_grid):
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    combos = [dict(zip(keys, values)) for values in product(*value_lists)]
    return combos


def log_likelihood(mu, y, param_dict, dist="norm"):
    likelihood = 0.0
    if dist == "student":
        # Student-t likelihood for Score filter
        nu = param_dict["nu"]
        sigma = param_dict["sigma"]
        for y_t, mu_t in zip(y, mu):
            # Compute log-likelihood of observation given state estimate
            ll_t = t.logpdf(y_t - mu_t, df=nu, scale=sigma)
            likelihood += ll_t
    elif dist == "norm":
        # Gaussian likelihood for Kalman filter
        residuals = np.array(y) - np.array(mu)
        # If sigma isn't provided, use the MLE estimate (standard deviation of residuals)
        sigma = param_dict["sigma"]
        likelihood = np.sum(norm.logpdf(residuals, scale=sigma))
    else:
        raise NameError

    return likelihood


def grid_search(y, mu_0, param_dicts, func, dist="norm"):
    best_params = None
    best_ll = -float("inf")
    for param_dict in param_dicts:
        # Create filter with current parameters
        filter = Filter(mu_0, param_dict=param_dict, func=func)
        filter.update(y)
        # Compute log-likelihood using one-step-ahead priors (mu_prior)
        # aligned with observations for p(y_t | y_{1:t-1})
        ll = log_likelihood(filter.mu_prior, y, param_dict, dist=dist)
        if ll > best_ll:
            best_ll = ll
            best_params = param_dict.copy()
    return best_params

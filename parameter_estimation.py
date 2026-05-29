from itertools import product

from scipy.stats import t

from robust_filter import Filter


def generate_param_grid(param_grid):
    keys = list(param_grid.keys())
    value_lists = [list(param_grid[k]) for k in keys]
    combos = [dict(zip(keys, values)) for values in product(*value_lists)]
    return combos


def log_likelihood(mu, y, param_dict):
    nu = param_dict["nu"]
    sigma = param_dict["sigma"]
    likelihood = 0.0
    for y_t, mu_t in zip(y, mu):
        # Compute log-likelihood of observation given state estimate
        ll_t = t.logpdf(y_t - mu_t, df=nu, scale=sigma)
        likelihood += ll_t
    return likelihood


def grid_search(y, mu_0, param_dicts, func):
    best_params = None
    best_ll = -float("inf")
    for param_dict in param_dicts:
        # Create filter with current parameters
        filter = Filter(mu_0, param_dict=param_dict, func=func)
        filter.update(y)
        # Compute log-likelihood of observations given filter estimates
        ll = log_likelihood(filter.mu, y, param_dict)
        if ll > best_ll:
            best_ll = ll
            best_params = param_dict.copy()
    return best_params

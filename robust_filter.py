def score_func(y_t, mu_t, param_dict):
    return (y_t - mu_t) / (
        1 + ((y_t - mu_t) ** 2 / (param_dict["nu"] * param_dict["sigma"] ** 2))
    )


def kalman_func(y_t, mu_t, param_dict):
    return y_t - mu_t


class Filter:
    def __init__(self, mu_0, param_dict, func=kalman_func):
        self.mu_0 = mu_0
        self.param_dict = param_dict
        self.func = func
        self.mu = None
        self.x = None
        self.mu_prior = None

    def update(self, y):
        """
        Update filter with observations.

        States:
        - mu_t: predicted/filtered state estimate at time t (one-step-ahead prior)
        - x_t: innovation (measurement residual) at time t
        - mu_prior: one-step-ahead predictions (aligned with y for likelihood)

        Update equation: mu_{t+1} = theta * mu_t + kappa * x_t
        where x_t = func(y_t, mu_t, param_dict)
        """
        self.mu = []
        self.x = []
        self.mu_prior = []
        mu_t = self.mu_0

        for y_t in y:
            # Store the prior (one-step-ahead prediction) before using y_t
            self.mu_prior.append(mu_t)

            # Compute innovation (residual)
            x_t = self.func(y_t, mu_t, self.param_dict)
            self.x.append(x_t)

            # Update state estimate: weighted combination of previous state and innovation
            mu_next = self.param_dict["phi"] * mu_t + self.param_dict["kappa"] * x_t
            self.mu.append(mu_next)

            # Propagate state for next iteration
            mu_t = mu_next

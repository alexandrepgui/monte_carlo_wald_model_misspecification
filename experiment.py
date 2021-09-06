import numpy as np
import pandas as pd
import math
from scipy.stats import chi2


def generate_sample(n):
    return np.absolute(np.random.normal(0, 1, size=n))


def wald_test(sample):
    n = len(sample)
    # The maximum likelihood estimator for theta is the inverse of the sample mean
    theta_n = 1 / sample.mean()
    # The minimizer of the Kullback-Leibler distance of the exponential pdf with respect to the half-normal pdf 
    theta_star = (math.pi/2) ** .5
    # Computing the A and B estimators
    A = sample.mean() ** 2
    B = sample.var(ddof=0)
    p = []
    for error_type in [1, 2]:
        if error_type == 1:
            # Computing s(theta_n) = theta_n - theta_star (The null hypothesis is correct)
            s = theta_n - theta_star
        elif error_type == 2:
            # Computing s(theta_n) = theta_n - 1.1 * theta_star (The null hypothesis is wrong)
            s = theta_n - 1.1 * theta_star
        # Computing the Wald statistic
        T_w = n * (s**2) * (A**2) / B
        # Computing p-values
        p.append(1 - chi2.cdf(T_w, 1))
    return theta_n, p[0], p[1]


def monte_carlo(sizes=[10, 50, 100, 200, 500], alphas=[.01, .05, .1], N=5000):
    for n in sizes:
        for i in range(N):
            # generates a n-sized sample
            sample = generate_sample(n)
            # performs Wald test
            theta, p_1, p_2 = wald_test(sample)
            results = {'n': n, 'theta': theta, 'p1': p_1, 'p2': p_2}
            for alpha in alphas:
                # Type I error if the null hypothesis is rejected
                results[f't1_error_{int(100*alpha)}%'] = (p_1 < alpha)
                # Type I error if the null hypothesis is accepted
                results[f't2_error_{int(100*alpha)}%'] = (p_2 > alpha)
            try:
                df = df.append(pd.Series(results).to_frame().T)
            except NameError:
                df = pd.Series(results).to_frame().T
    order_cols = ['n', 'theta', 'p1', 'p2'] + [f't1_error_{int(100*alpha)}%' for alpha in alphas] + [f't2_error_{int(100*alpha)}%' for alpha in alphas]
    return df[order_cols]


def wrapper():
    np.random.seed(10)
    df = monte_carlo()
    aggs = {i: [np.mean, np.std] for i in ['theta', 'p1', 'p2']}
    aggs.update({f't{i}_error_{int(100*alpha)}%': np.mean for i in [1, 2] for alpha in [.01, 0.05, .1]})
    df = df.astype(float).groupby(by='n').agg(aggs)
    df.to_csv('experiment_results.csv')
    print(df.T)


if __name__ == '__main__':
    wrapper()
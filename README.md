# monte_carlo_wald_model_misspecification
A Monte Carlo experiment for a Wald Test under the assumption of model misspecification. True distribution is half-normal, hypothesized model is exponential.

The Wald statistic is the one suggested in Theorem 3.4 (White, 1982), with the function $s$ being

$$
s(\theta) = \theta - \theta_*
$$

for identification of Type I error, and

$$
s(\theta) = \theta - 1.1\theta_*
$$

for identification of Type II error.

References:

Maximum Likelihood Estimation of Misspecified Models, White 1982

Lecture 16 — MLE under model misspecification, STATS 200: Introduction to Statistical Inference - Stanford

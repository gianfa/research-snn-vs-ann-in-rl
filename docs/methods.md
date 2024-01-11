# Methods

- [References](#references)
- [Evaluation](#evaluation)

## References

1. [MAASS, Wolfgang; NATSCHLÄGER, Thomas; MARKRAM, Henry. Real-time computing without stable states: A new framework for neural computation based on perturbations. Neural computation, 2002, 14.11: 2531-2560.](https://igi-web.tugraz.at/people/maass/psfiles/130.pdf)
2. [PONGHIRAN, Wachirawit; SRINIVASAN, Gopalakrishnan; ROY, Kaushik. Reinforcement learning with low-complexity liquid state machines. Frontiers in Neuroscience, 2019, 13: 883.](https://www.frontiersin.org/articles/10.3389/fnins.2019.00883/full)

## Evaluation

In order to evaluate the performance of the model, the following metrics, implemented in scikit-learn, were adopted in cross validation.

- [mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error). The lower the better.
- [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). The lower the better.
- [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) (coefficient of determination, $R^2$ ). The higher the better. It is a computationally robust version, for which perfect predictions score 1 and imperfect predictions score 0.

$$
R^2(y, \hat{y})=1-\frac{ \sum^{n}_{i=1} (y_i - \hat{y_i})^2}{\sum^{n}_{i=1} (y_i - \bar{y_i})^2}
$$
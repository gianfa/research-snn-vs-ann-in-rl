# Forecast of Lorenz signal with ESN \[Sketch\]

- [Evaluation](#evaluation)
- [References](#references)

## Evaluation

In order to evaluate the performance of the model, the following metrics, implemented in scikit-learn, were adopted in cross validation.

- [mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error). The lower the better.
- [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). The lower the better.
- [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) (coefficient of determination, $R^2$ ). The higher the better. It is a computationally robust version, for which perfect predictions score 1 and imperfect predictions score 0.  
$$
R^2(y, \hat{y})=1-\frac{ \sum^{n}_{i=1} (y_i - \hat{y_i})^2}{\sum^{n}_{i=1} (y_i - \bar{y_i})^2}
$$

## References

1. https://scikit-learn.org
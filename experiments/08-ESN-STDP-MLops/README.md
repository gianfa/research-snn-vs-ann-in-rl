# Forecast of Lorenz signal with ESN \[Sketch\]

- [Evaluation](#evaluation)
- [References](#references)
  - [Notes](#notes)
  - [TODO](#todo)
  - [RQ](#rq)

## Evaluation

In order to evaluate the performance of the model, the following metrics, implemented in scikit-learn, were adopted in cross validation.
* [mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error). The lower the better.
* [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html). The lower the better.
* [r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) (coefficient of determination, $R^2$ ). The higher the better. It is a computationally robust version, for which perfect predictions score 1 and imperfect predictions score 0.  
$$
R^2(y, \hat{y})=1-\frac{ \sum^{n}_{i=1} (y_i - \hat{y_i})^2}{\sum^{n}_{i=1} (y_i - \bar{y_i})^2}
$$

## References

1. https://scikit-learn.org
2. 

### Notes

### TODO

- v track eigenvalues during STDP
- v visualize evolution vs weights/eigenvalues
- visualize
    - v shift vs forecast
    - v shift vs performance



characterize

### RQ
1. Does the STDP amplify performance for constrained reservoirs (vs Resv size)?
2. How about different STDP?
3. Spiking:
    1. what's the relationship between plasticity and encoding?
    2. performance vs neuron type

tanh(
    {
        tanh(sum(w * input_a) + 
    } + tanh(sum(w * input_b) ...
)



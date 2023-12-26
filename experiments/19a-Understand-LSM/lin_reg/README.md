# Signal detection step by step

- [The task](#the-task)
- [The Data](#the-data)
  - [Score](#score)
- [The experiments](#the-experiments)
  - [01. Simple LogReg](#01-simple-logreg)
  - [02. LogReg with feature extraction](#02-logreg-with-feature-extraction)

## The task

> Detection theory or signal detection theory is a means to measure the ability to differentiate between information-bearing patterns (called stimulus in living organisms, signal in machines) and random patterns that distract from the information (called noise, consisting of background stimuli and random activity of the detection machine and of the nervous system of the operator). [Detection Theroy, Wikipedia](https://en.wikipedia.org/wiki/Detection_theory)

Classification of frequency pattern in a signal.

## The Data

A signal made by a baseline and some segments characterized by a frequency pattern.

Labels are 1 where the frequency pattern segment is and 0 elsewhere.

## The Estimator

### Score
Mean accuracy, see [sklearn.linear_model.LogisticRegression.score](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.score)

## The experiments

balance_train: 0.42

### 01. Simple LogReg

> clf Score on X_train pca 0.98
> clf Score on X_test pca 0.98

<img src='../imgs/01-logreg-scaled.png'>

### 02. LogReg with feature extraction

> clf Score on X_train pca 0.80
> clf Score on X_test pca 0.90

<img src='../imgs/02-logreg-pca.png'>



# Signal detection step by step

- [The task](#the-task)
- [The Data](#the-data)
- [The Estimator](#the-estimator)
  - [Score](#score)
- [The experiments](#the-experiments)
  - [01. Simple LogReg](#01-simple-logreg)
  - [02. LogReg with feature extraction](#02-logreg-with-feature-extraction)
  - [03. Multinomial LogReg](#03-multinomial-logreg)
  - [04. Multinomial LogReg with longer window size](#04-multinomial-logreg-with-longer-window-size)

## The task

> Detection theory or signal detection theory is a means to measure the ability to differentiate between information-bearing patterns (called stimulus in living organisms, signal in machines) and random patterns that distract from the information (called noise, consisting of background stimuli and random activity of the detection machine and of the nervous system of the operator). [Detection Theroy, Wikipedia](https://en.wikipedia.org/wiki/Detection_theory)

Classification of frequency pattern in a signal.

## The Data

A signal made by a baseline and some segments characterized by a frequency pattern.

Labels are 1 where the frequency pattern segment is and 0 elsewhere.

## The Estimator

### Score

Mean accuracy, see [sklearn.linear_model.LogisticRegression.score](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.score)

## The experiments

balance_train: 0.42

### 01. Simple LogReg

```mermaid
graph LR
  split[Train/Test split];
  fit[LogReg Fitting]
  pred[Prediction]

  A[Data Generation] --> split
  split --> fit
  fit --> pred
```

- > clf Score on X_train pca 0.98
- > clf Score on X_test pca 0.98

<img src='../imgs/01-logreg-scaled.png'>

### 02. LogReg with feature extraction

```mermaid
graph LR
  split[Train/Test split];
  pca[PCA transformation on train and test];
  fit[LogReg Fitting]
  pred[Prediction]

  A[Data Generation] --> split
  split --> pca
  pca --> fit
  fit --> pred
```

- > clf Score on X_train pca 0.80
- > clf Score on X_test pca 0.90

<img src='../imgs/02-logreg-pca.png'>

### 03. Multinomial LogReg

- 3 categories: baseline (constant), freq 1, freq2
- window size: 10

```mermaid
graph LR
  split[Train/Test split];
  fit[LogReg Fitting]
  pred[Prediction]

  A[Data Generation] --> split
  split --> fit
  fit --> pred
```

It can be seen that the model struggles to distinguish all three categories; rather, it tends to collect everything outside the baseline into a single category.

<img src='../imgs/03-multi_logreg-signal.png'>

<img src='../imgs/03-multi_logreg-predictions.png'>

<img src='../imgs/03-multi_logreg-pred_by_cat-comparison.png'>  

### 04. Multinomial LogReg with longer window size

- 3 categories: baseline (constant), freq 1, freq2
- window size: 200

```mermaid
graph LR
  split[Train/Test split];
  fit[LogReg Fitting]
  pred[Prediction]

  A[Data Generation] --> split
  split --> fit
  fit --> pred
```

Extending the window size shows that the model begins to distinguish all three categories; albeit with its own limitations.

<img src='../imgs/04-multi_logreg-feateng_200-signal.png'>

<img src='../imgs/04-multi_logreg-feateng_200-predictions.png'>

<img src='../imgs/04-multi_logreg-feateng_200-pred_by_cat-comparison.png'>
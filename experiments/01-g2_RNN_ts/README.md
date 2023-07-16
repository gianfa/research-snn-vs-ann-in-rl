# 2nd Gen. timeseries Classification

Description
-----------
We want to classify the original phases of a given signal.
X -> RNN -> y

- Such a signal is a function of two phases w1, w2: X_i = s(w1, w2); w1 << w2
- The labels are such phases, encoded as one-hot arrays: y = 1h(w1, w2)

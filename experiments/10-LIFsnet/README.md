# Training a recurrent LIF network

In this experiment, an attempt is made to train a recurrent network of LIFs to recognise artifacts in an input signal and to understand what an optimal strategy for doing so might be. 

In particular, we want to investigate the trainability of the following architecture. 

- Single LIF layer 
  - Composed of a number of LIFs greater than or equal to 1.
  - All LIFs have with the same characteristics.
- Recurrent connections limited in states, $S = \{s_{0,1}, .. s_{i, j}\}; s_{i,j} \in \{0, 0.5, 1\}$



## Estimator

An additional LIF layer has been added, as an output one. 
This receives at each end of the inner loop the activations from the hidden layer. 
Hidden layer and output layer are connected by a matrix of statistical weights
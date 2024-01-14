# STDP in 2nd gen: Fully Connected Network

In this experiment, STDP is implemented in a fully connected network

The process is summarised by the implementation of the following steps. Given a FC neural network:

1. Saving of **layer activations** at each forward step of the network. In the case of mini-batch training, the activations of the whole batch are saved layer by layer
2. Conversion of activations into **spike traces**
3. **STDP step** implementation, with weight update between layers

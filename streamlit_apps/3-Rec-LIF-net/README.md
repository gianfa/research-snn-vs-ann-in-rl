# Recurrent LIF netowrk


## How to make it work

Just move to the *streamlit_apps* directory and execute the following command

```bash

streamlit run 3-Rec-LIF-net/app.py
```


## How it works

In this experiment, a recurrent neural network is implemented, made up of a single LIF layer, which receives an input current. 

The setup is very simple. 

At each time step t:  
the input current $I_{in}$ is fed according to the weights expressed in the matrix $W_{LIF, in}$ for the adjacency matrix $A_{LIF, in}$. Before doing so, 
the LIF layer activations of the previous instant are added to the input current to be passed as total input to the LIF. 

The LIF activations at a given instant are passed to the next instant according to the connections expressed in the adjacency matrix $A_{LIF, LIF}$ and according to the weights of the linear layer $W_{LIF, LIF}$.


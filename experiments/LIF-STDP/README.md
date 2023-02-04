# STDP Implementation trials, on LIFs

Description
----------
Contains tests of STDP done on LIF networks.

Notes
-----
Here we need to solve the issue about the dynamics of the network with 
respect to the time of administration of the stimulus (seeing the example)
and the time of the dynamics of the neuron.
Coming from the surGrad, the training takes place:
1. for each example
2. LIFs are initialized
3.The dynamics of the LIFs is left to evolve by a delta t.  

This is a problem because the dynamics are reset for each sample, eliminating 
the "life history" of the neurons, which is useful in the STDP regime. 


References
----------
1. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html#define-the-network
2. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
3. `NEFTCI, Emre O.; MOSTAFA, Hesham; ZENKE, Friedemann. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 2019, 36.6: 51-63. <https://core.ac.uk/download/pdf/323310555.pdf>`_


Theory
------
Learning: Loss
- Surrogate gradient approach.
- Spike-Operator approach. -> ∂S/∂U ϵ {0,1}


Output decoding
- we take the output layer as a logit layer, where each neuron
    fires for a specific class.
- we choose "rate coding" as a decoding strategy. This means that
    we'll expect the neuron assigned to the right class to fire
    at higher freq during the specific class showing up.
    - ONE way is to increase $U_th$ of the neuron and decrease the others ones
        - $softmax p_i[t] = exp(U_i[t] / sum( exp(U_j[t]), 0, C))$
        - $L_{CE}[t] = sum( y_i log(p_i[t]), 0,C )$
        - $=> L_{CE} = sum( L_{CE}[t] ,t)$
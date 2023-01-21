
# Contents
* [Conventions](#conventions)
* [References](#References)
* [Training stage](#train)
* [Questions](#qs)
* [Assumptions](#assumptions)
* [Nice to study](#tostudy)
* [Notes on proj steps](#proj_steps)


<div id='conventions'>Conventions</div>

[Maas, 1996](https://igi-web.tugraz.at/people/maass/psfiles/85a.pdf)
* 2GNN: 2nd generation neural network
* 3GNN: 3nd generation neural network

<div id='ref'>References</div>
* https://deepai.org/publication/stdp-learning-of-image-patches-with-convolutional-spiking-neural-networks

* [STDP, spikingjelly, official docs](https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/stdp.html)
* [learning.py, spikingjelly, github](https://github.com/fangwei123456/spikingjelly/blob/d9b8934ede19cbdbb44c139592c4ee6b24cda84d/spikingjelly/activation_based/learning.py)

https://compneuro.neuromatch.io/tutorials/W2D3_BiologicalNeuronModels/student/W2D3_Tutorial4.html

* [Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling](https://www.izhikevich.org/publications/dastdp.pdf)


* https://github.com/BrainCog-X/Brain-Cog/issues/26 


<div id='train'>Training stage</div>

STDP
* is a sort of optimization -> optimizer
* stop rule:
    - `dw.mean() <= w.mean()*0.05` for 10 times
    - `max iterations = 200`

<div id='assumptions'>Assumptions</div>
- STDP is disjointed from typical 2GNN training
- same time granularity across all the network
    - each time step is a network step, thus is a STDP step
- STDP is a optimizer
    - based on the STDP formula
- a loss is needed. It would be based on dw variation in time. Therefore:
    - loss = 
        - (dw_in - dw_out)/dw_in < th # fraction based
        - ( (dw_in - dw_out)/dw_in < th )[-last_steps:]/last_steps # avg frac based
            - needs a internal time buffer
- there are two time scales: the receptors one and the inner STDP neurons one.
- W matrix is not enough for pre-post interaction. An adjacency matrix is needed. One can argue that two consecutive layers might be enough to create a Pre vs Post distinction (W: prex post). However, this might represent only a feedforward, not a possibly recursive network. [need to review].
    - We want to make room for synaptic properties:
        - strength
        - directionality (pre-post)
        - inhibition/excitation
- A single layer of neurons, connected via an Adj. matrix, is preferred. Reasoning by successive layers, imposes a global directionality on the sequence of spikes, which is not always desirable.
- There are two different moments:
    1. Update neurons membranes (activations)
    2. Udate weights (STDP)

Sketch
------

```python
dt = 1e-3 # s
t_range = 100 # s
num_steps = int(t_range/dt)

lif1 = snn.Lapicque(
    R=R, C=C, time_step=dt, threshold=0.5,
    reset_mechanism="zero")
for step in range(num_steps):
    spk_out, mem = lif1(cur_in[step], mem)
    mem_rec.append(mem) # record states
    spk_rec.append(spk_out)
```

* input(100), Leaky(), hidden(70), Leaky(); but it is not perfectly clear to me how Leaky can act as a activation function for all of the above and generate a rasterplot of the input size.

Bulding blocks:
- dw(dt) function
- spikes collection rules
- dw rules

<div id='qs'>Questions</div>

1. STDP Optimizer relies on STDP parameters.
    - Can they be learned?
    - What's the added value?

2. Does STDP act synchronously or asynchronously with respect to the stimulation of the agent, i.e., the entire network?
* sol1. They are sync.
    * opt1. update as stimulus comes. Everytime a neighbour spikes after having received one, weights are updated.
    * opt2. update time batches. Are effects separable?
* sol2. Two time scales: one for stimulus (x reception) and one for STDP. If they are equal, then the processes are sync.

3. The update phase of STDP weights covers peaks, weights, and connections. If we consider peaks to be the result of activations, we can also consider STDP in a domain of 2ndGen networks.
Using it, however, there are some points to take into account:
- Based on the activation distance of ....

4. Tut 5 he uses init_leaky(), meaning mem=[], in a forward step.
Does this make sense?

Ex.
200 steps

traces = []
for step in steps:
    traces.append(forward)
    if step % 5 == 0:
        STDP_step



### <div id='tostudy'>Nice to study</div>

* Comparisons
    * computational efficiency comparison applying decomposition to datasets, by:
        - PCA
        - Autoencoder
        - STDP
* Neural backpropagation
    * >https://en.wikipedia.org/wiki/Neural_backpropagation#Algorithm
* Neuron models
    * LI&delayied
        * >https://neuronaldynamics.epfl.ch/online/Ch6.html
* STDP rules
    * Triplet rule
        * >http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Triplet_rule_of_STDP

#### To tidy

What does work?
* topology
* weight init
* neuron parameters
* plasticity
* refractory?


* noisy input


* regole "direzionalità" connessioni (assoniche?)

* study **minimal working topology**
    * examples
        1. spatial distribution of neurons.


* STDP cost function?

* insect eye
    * convolutional kernel pyramid


FC(N1, N2)
FC(N2, N3)
FC(N4, N5)


<div id='proj_steps'>Notes on proj steps</div>
------------------------
1. Hypothesys formulation: "What if STDP is implemented in snnTorch and we can build mixed DNN?"
2. Search for code solutions in the web and literature.
    * Rather simple implementations, with few literary references.
    * Some particularly sophisticated implementations.
3. Review of pytorch and stages of deep learning.
    * Custom loss
    * Custom optimizer
4. First solution formulation and code draft.
    * STDP as an optimisation process in SL.
    * STDP stop rule.
5. Further study in literature
    * Identification of the theoretical basis of simple implementations. (Gerstner, Abbott, Izhikevich).
6. First working implementation
    * A 2nd gen DL version that implements STDP as an optimizer and loss
7. NEXT: study how to connect snnTorch SpikingNeuron to STDP & implement STDP update over the network
8. refactory for a more flexible structure, in order to implement
    many rules.
9. 


## NEXT
* echo state network (STDP)
    * IN -> reservoir -> OUT
* [Modeling neural plasticity in echo state networks for classification and regression](https://sci-hub.st/https://doi.org/10.1016/j.ins.2015.11.017)
    * Diversi task
        * Hebbian/non-Hebbian rule
        * miglioramento performance proporzionale al t di esposizione
    * > Riprendere paper per riordinare
        * > estendere formalismo?
* Synthetic Dataset
* SNN
    * how to make a neuron a decisor?
        * maybe as a multimodal/multithresholded neuron?
            * there are all types neurons in the body, so maybe it can be a moving threshold neuron (driven by freq?)
        * reservoir + decisor neuron?

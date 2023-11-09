# Topological Analysis

## Contents
- [Contents](#contents)
  - [The Signal](#the-signal)
- [The Network](#the-network)
  - [The Reservoir](#the-reservoir)
  - [The Readout](#the-readout)
- [The Training](#the-training)
  - [The Loss Computation](#the-loss-computation)
- [Experiment](#experiment)
- [Visualization](#visualization)
  - [Top n connections by performance](#top-n-connections-by-performance)
    - [Graph of the Top n connections by performance](#graph-of-the-top-n-connections-by-performance)
  - [Mean performance by shift wrt diagonal](#mean-performance-by-shift-wrt-diagonal)
- [Back to LSM Topological Analysis](#back-to-lsm-topological-analysis)

### The Signal
The signal on which the network was trained consists of a constant baseline and alternating fragments whose frequency can be one of two possible

## The Network

<img src='./imgs/network1.png'>

The neural network is a Liquid State Machine (LSM) \[Maas, 2002\], with the following characteristics.

### The Reservoir

* Total number of neurons: 20
* Number of input neurons: 10
* Number of output neurons: 10
* Weights values randomly assigned (Normal)
* Synaptic connections with decay
   * Randomly distributed (Normal)
   * decay function: $g[t+1] = q \cdot g[t]$, with $g$ synaptic gain (conductance) and $q \in [0, 1]$ decay rate.
   * decay rate: .25

### The Readout

Fully connected layer.
* Number of neurons equal to the number of categories to classify

## The Training

* Optimizer: Adam \[Kigma, 2014\]
* Loss based on CrossEntropy

At each training step the network receives a signal element.
The network subsequently completes an entire propagation step between the layers until a prediction is reached.  
A buffer of a specific size, sequentially collects a specific amount of network outputs.

### The Loss Computation
At a specific step interval, $l_{scope}$ (250), the loss is calculated over the the whole buffer.


## Experiment
After researching the most important hyperparameters,
we chose a configuration to analyze.  
* reservoir_size: 20,
* radius: 4,
* degree in \{2, 3\}.

Comprehensive training was repeatedly launched and performance was collected.  

100 runs were performed for each change in (degree, radius).

The performance metrics chosen here is the average prediction accuracy in the last 10% portion of the signal, $acc_{10p}$.


## Visualization
Below we see a representation of the average performance obtained by connection position on the adjacency matrix.

<img src='./topological_analysis/perf_x_pos-mean.png'>
<!-- Each graph was obtained by multiplying the binary adjacency matrix of each run by $acc_{10p}$ and finally averaging for each cell.

<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_1-r_4.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_2-r_4.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_3-r_4.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_4-r_4.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_3-r_3.png' width=50%> -->


<!-- Here below we see reported the same values as in the graphs above, but filtered by $acc_{10p}$ > 0.25

<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_1-r_4-25pc.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_2-r_4-25pc.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_3-r_4-25pc.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_4-r_4-25pc.png' width=50%>
<img src='./topological_analysis/avg_acc_per_pos-ressize_20-d_3-r_3-25pc.png' width=50%> -->

### Top n connections by performance

Here below we see the top connections by mean performance

<img src='./topological_analysis/top_perf_x_pos-mean.png'>

#### Graph of the Top n connections by performance

Below we see a graph rapresentation of the top neuron positions

<img src='./topological_analysis/top_perf_graph.png'>

### Mean performance by shift wrt diagonal

<img src='./topological_analysis/shift-perf_x_pos-mean.png'>

<img src='./topological_analysis/shift-top_perf_graph.png'>

<img src='./topological_analysis/shift-top_perf_x_pos-mean.png'>


## Back to LSM Topological Analysis

[LSM-topological_analysis](./../../../docs/LSM-topological_analysis.md)

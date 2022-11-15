"""
ref1: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb#scrollTo=e4LPD0WCobYx
ref: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html

"""
# %%

DATA = '../_data/'


# %%

import snntorch as snn  # noqa
import torch  # noqa

# Training Parameters
batch_size = 128
data_path = DATA
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

# %% Download datasetSNN

from torchvision import datasets, transforms  # noqa

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(
    data_path, train=True, download=True, transform=transform)

print(mnist_train.train_data.shape)

# %% Reduce the training set to i/10

from snntorch import utils  # noqa

subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")

# %% Create DataLoaders for passing data in batches

from torch.utils.data import DataLoader  # noqa

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

# %% Spike Encoding
"""
The module snntorch.spikegen (i.e., spike generation) contains a series of
    functions that simplify the conversion of data into spikes.
    There are currently three options available for spike encoding
    in snntorch:
    * Rate coding: spikegen.rate (f)
    * Latency coding: spikegen.latency (t)
    * Delta modulation: spikegen.delta (dt)

How do these differ?
Rate coding uses input features to determine spiking frequency
Latency coding uses input features to determine spike timing
Delta modulation uses the temporal change of input features to generate spikes

"""
from snntorch import spikegen  # noqa

# %% ## Rate coding
""" (grey tone) -> (firing probab)
Given the normalised samples X_i,

Take the normalised input feature X_ij as a probability.
It is the probability that a spike will occur at any given time step, thus
it's a rate-coded value, R_ij.

This can be treated as a Bernoulli trial: R_ij ~ B(n, p), with:
* n, number of trials, equal to 1
* p, probability of spiking, p = X_ij.


A white pixel corresponds to a 100% probability of spiking,
 and a black pixel will never generate a spike.
"""

# Temporal Dynamics
num_steps = 10  # for n -> ∞ the success rate tends to p

# create vector filled with 0.5
raw_vector = torch.ones(num_steps) * 0.5

# pass each sample through a Bernoulli trial
rate_coded_vector = torch.bernoulli(raw_vector)
print(f"Converted vector: {rate_coded_vector}")

spike_rate = rate_coded_vector.sum()*100/len(rate_coded_vector)
print(f"The output is spiking {spike_rate:.2f}% of the time.")


# %% using spikegen.rate
"""
If the input falls outside of  [0,1],
    this no longer represents a probability.
Such cases are automatically clipped to ensure the feature
    represents a probability.

The structure of the input data is
    [num_steps x batch_size x input dimensions]:
"""

from snntorch import spikegen  # noqa

# Iterate through minibatches
data = iter(train_loader)
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)
print(spike_data.size())
print(
    "n_steps: ", str(spike_data.size()[0]),
    "\nbsize:" + str(spike_data.size()[1]),
    "\ninput_dim:" + str(spike_data.size()[1:]),
)


# %% Visualization
import matplotlib.pyplot as plt  # noqa
import snntorch.spikeplot as splt  # noqa
from IPython.display import HTML  # noqa


# %%
"""
To plot one sample of data,
 index into a single sample from the batch (B
 dimension of spike_data, [T x B x 1 x 28 x 28]:

"""

spike_data_sample = spike_data[:, 0, 0]  # all T, B=0, sample=0
print(spike_data_sample.size())

# %%
"""
$ brew install ffmpeg
$ brew info ffmpeg # -> /usr/local/...../ffmpeg
"""
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/Cellar/ffmpeg/5.1.1/bin/ffmpeg'

HTML(anim.to_html5_video())
# anim.save("spike_mnist_test.mp4")
print(f"The corresponding target is: {targets_it[0]}")

# %%
"""
MNIST features a greyscale image, and the white text
guarantees a 100% of spiking at every time step.
So let's do that again but reduce the spiking frequency.
This can be achieved by setting the argument `gain`.
Here, we will reduce spiking frequency to 25%.
"""
spike_data = spikegen.rate(
    data_it, num_steps=num_steps, gain=0.25)

spike_data_sample2 = spike_data[:, 0, 0]
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample2, fig, ax)
HTML(anim.to_html5_video())

# %%
# fig, ax = plt.subplots(1, 3)
plt.figure(facecolor="w")
plt.subplot(1, 3, 1)
plt.imshow(mnist_train.train_data[0], cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(spike_data_sample.mean(axis=0).reshape((28, -1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 1')

plt.subplot(1, 3, 3)
plt.imshow(spike_data_sample2.mean(axis=0).reshape((28, -1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 0.25')

plt.show()
# %% Raster Plots
"""
[```snntorch.spikeplot.raster(data, ax, **kwargs)```](https://snntorch.readthedocs.io/en/latest/snntorch.spikeplot.html#snntorch.spikeplot.raster)
This requires reshaping the sample into a 2-D tensor, where 'time' is the first dimension. Pass this sample into the function spikeplot.raster.
"""
print(
    "Reshaping (flattening) over the time steps, we'll obtain:\n" +
    f"{spike_data_sample2.shape} -> " +
    f"{spike_data_sample2.reshape(10, -1).shape}"
)

# Reshape
spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))

# raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample2, ax, s=1.5, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# %%

idx = 210  # index into 210th neuron

fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

(splt
    .raster(
        spike_data_sample
        .reshape(num_steps, -1)[:, idx]
        .unsqueeze(1),
        ax, s=100, c="black", marker="|"))

plt.title(f"Input Neuron {idx}")
plt.xlabel("Time step")
plt.yticks([])
plt.show()
# %%
"""
Notes. The rate coding appears to be a sort of generative modeling, since it
    is based on shaping a probability function pixel-wise, then you can
    actually use such array of prob functions in order to generate new data.
"""

# %% ## Latency coding
""" (grey tone) -> (t step)
    Temporal codes capture information about the precise firing time of
neurons; a single spike carries much more meaning than in rate codes which
rely on firing frequency.
While this opens up more susceptibility to noise,
it can also decrease the power consumed by the hardware running SNN
algorithms by orders of magnitude.

    ```spikegen.latency()``` allows each input to fire at most once during
the full time sweep. Features closer to 1 will fire earlier and features
closer to 0 will fire later. I.e., in our MNIST case, bright pixels will fire
earlier and dark pixels will fire later.

All that matters is: big input means fast spike; small input means late spike.

I_in = I_R + I_C
I_in = V(t)/R + dV(t)/dt
V(t) = I_in R + c_1 e^{-t/(R C)}

t=0 & V=0 => c_1 = -I_in R
=> V(t) = -I_in R (1 - e^{-t/(R C)})

then we can compute the time taken for V(t) to go from 0 to
a V_th as a function t(I_in; V_th, tau)
t = tau[ ln (I_in /(I_in - V_th)) ]
"""

# %%
"""
Convert X_ij ∈ [0,1] -> L_ij

t = tau[ ln (I_in /(I_in - V_th)) ]
"""


def convert_to_time(data, tau=5, threshold=0.01):
    spike_time = tau * torch.log(data / (data - threshold))
    return spike_time


for tau_i in range(0, 20, 5):
    raw_input = torch.arange(0, 5, 0.05)  # [A] tensor from 0 to 5
    spike_times = convert_to_time(raw_input, tau=tau_i)

    plt.plot(
        raw_input, spike_times,
        label=f"tau={tau_i}")
    plt.xlabel('Input Value')
    plt.ylabel('Spike Time (s)')
    plt.legend()
plt.show()

# %%
"""
We can automate
raw_input -> spike_time -> spike, by

```spikegen.latency```

"""
spike_times = convert_to_time(raw_input)
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01)

print(type(spike_data))
print(spike_data.shape)

# %% raster plot

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].reshape(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# optional save
# fig.savefig('destination_path.png', format='png', dpi=300)
# %%
"""
   The logarithmic code coupled with the lack of diverse input values
(i.e., the lack of midtone/grayscale features) causes significant clustering
in two areas of the plot.
The bright pixels induce firing at the start of the run, and the dark pixels
at the end. We can increase tau to slow down the spike times, or linearize
the spike times by setting the optional argument linear=True.
"""

spike_data = spikegen.latency(
    data_it, num_steps=100, tau=5, threshold=0.01, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].reshape(num_steps, -1), ax, s=25, c="black")
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
# %%
"""
   It is notably happening all around the first steps,
so we can relax the decay, increasing tau.
Another option is to span the full range of the steps,
selecting `normalize=True`.
"""
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01,
                              normalize=True, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].reshape(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
# %%
"""
    One major advantage of latency coding over rate coding is sparsity.
If neurons are constrained to firing a maximum of once over the time course
of interest, then this promotes low-power operation.
In the scenario shown above, a majority of the spikes occur at the
final time step, where the input features fall below the threshold.
    In a sense, the dark background of the MNIST sample holds no useful
information. We can remove these redundant features by setting `clip=True`.
"""
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, 
                              clip=True, normalize=True, linear=True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].reshape(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
# %% Animation

spike_data_sample = spike_data[:, 0, 0]
print(spike_data_sample.size())

fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)

HTML(anim.to_html5_video())

# Save output: .gif, .mp4 etc.
# anim.save("mnist_latency.gif")

print(targets_it[0])
"""
What we see:
1. t step 1: almost the whole number
2. t step 2 ....n : the rest of grey tones according to their value.

    This animation is obviously much tougher to make out in video form,
but a keen eye will be able to catch a glimpse of the initial frame
where most of the spikes occur.
Index into the corresponding target value to check its value.
"""
# %% ## Delta Modulation coding
"""
"biology is event-driven. Neurons thrive on change."

The `snntorch.delta` function accepts a time-series tensor as input.
It takes the difference between each subsequent feature across all time steps.
By default, if the difference is both positive and greater than the threshold
V_th , a spike is generated:


    Delta modulation is based on event-driven spiking.
The snntorch.delta function accepts a time-series tensor as input.
It takes the difference between each subsequent feature across all time
steps. By default, if the difference is both positive and greater than
the threshold  V_th, a spike is generated:
"""

# Create a tensor with some fake time-series data
data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])

# Plot the tensor
plt.plot(data)

plt.title("Some fake time-series data")
plt.xlabel("Time step")
plt.ylabel("Voltage (mV)")
plt.show()

# %%
"""
Pass the above tensor into the spikegen.delta function, with an arbitrarily
selected threshold=4
"""
# Convert data
spike_data = spikegen.delta(data, threshold=4)

# Create fig, ax
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

# Raster plot of delta converted data
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

# %%
"""
There are three time steps where the difference between  𝑑𝑎𝑡𝑎[𝑇]  and  𝑑𝑎𝑡𝑎[𝑇+1]  is greater than or equal to  𝑉𝑡ℎ𝑟=4 . This means there are three on-spikes.

The large dip to  −20  has not been captured above. If negative swings have importance in your data, you can enable the optional argument off_spike=True.
"""

# Convert data
spike_data = spikegen.delta(data, threshold=4, off_spike=True)

# Create fig, ax
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)

# Raster plot of delta converted data
splt.raster(spike_data, ax, c="black")

plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.xlim(0, len(data))
plt.show()

print(spike_data)



# %% ## Spike Generation

# Create a random spike train
spike_prob = torch.rand((num_steps, 28, 28), dtype=dtype) * 0.5
spike_rand = spikegen.rate_conv(spike_prob)

fig, ax = plt.subplots()
anim = splt.animator(spike_rand, fig, ax)

HTML(anim.to_html5_video())

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_rand[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# %%

""""""
import matplotlib.pyplot as plt
import numpy as np


def generate_raster_plot(spikes_tensor, title="Raster Plot"):
    """ Generate a raster plot from a spiking activity tensor.

     Parameters
    ----------
    spikes_tensor : numpy.ndarray
        A binary tensor where each row represents a neuron and each column is 0 or 1
        to indicate whether the neuron fired (1) or not (0).

    title : str, optional
        Title of the raster plot. Default is "Raster Plot".

    Returns
    -------
    None

    Example
    -------
    >>> num_neurons = 20
    >>> num_time_steps = 1000
    >>> spikes_tensor = np.random.choice([0, 1], size=(num_neurons, num_time_steps), p=[0.9, 0.1])
    >>> generate_raster_plot(spikes_tensor, title="Example Raster Plot")
    """
    num_neurons, num_time_steps = spikes_tensor.shape

    # Find indices where there are 1 values
    time_indices = [
        np.where(spk.flatten() == 1)[0]
        for ni, spk in enumerate(spikes_tensor)]

    # Generate the raster plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.eventplot(
        time_indices,
        lineoffsets=np.arange(len(time_indices)),
        linelengths=0.8, colors="black")
    ax.set_yticks(range(num_neurons))  # it messes up when too many neurons 
    ax.set_xlabel("Time")
    ax.set_ylabel("Neurons")
    ax.set_title(title)
    ax.set_xlim(0, num_time_steps)
    ax.set_ylim(0, num_neurons)
    ax.invert_yaxis()  # Invert the y-axis to have neuron 0 at the top
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax



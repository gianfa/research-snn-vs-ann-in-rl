""" Synapse

"""
import torch

class Synapse:
    """Synapse class

    Parameters
    ----------
    initial_conductance : _type_
        _description_
    decay_rate : _type_
        _description_
    learn_conductance : bool, optional
        _description_, by default False

    Example
    -------
    >>> size = 5
    >>> initial_conductance = 0.8
    >>> decay_rate = 0.9
    >>> synapse = Synapse((5, 5), initial_conductance, decay_rate)
    >>> spike_history1 = torch.tensor([[1], [0], [1], [1], [0]], dtype=torch.bool)
    >>> spike_history2 = torch.tensor([[0], [1], [1], [1], [1]], dtype=torch.bool)
    >>> print(spike_history)
    >>> synapse.update(spike_history1)
    >>> print(synapse.synaptic_matrix)
    >>> synapse.update(spike_history2)
    >>> print(synapse.synaptic_matrix)
    """
    def __init__(
        self,
        size: tuple,
        initial_conductance: torch.Tensor,
        decay_rate: torch.Tensor,
        recovery_rate: torch.Tensor = torch.tensor(.4),
        learn_conductance: bool = False):
        self.initial_conductance_ = initial_conductance
        self.synaptic_matrix = torch.ones(size) * initial_conductance
        assert decay_rate.dim() <= 2
        assert decay_rate.shape == (size[0], 1) \
            or decay_rate.shape == ()
        self.decay_rate = decay_rate

    def update(self, spk: torch.Tensor):
        
        # mask of the neurons who had a spike
        spiked_neurons = spk.view(-1).nonzero().squeeze()
        # synaptic decay for firing neurons
        if self.decay_rate.shape == ():
            self.synaptic_matrix[spiked_neurons] *= self.decay_rate
        else:
            self.synaptic_matrix = self.synaptic_matrix.T
            self.synaptic_matrix *= self.decay_rate
            self.synaptic_matrix = self.synaptic_matrix.T

        # reset conductance where not spiking
        size = self.synaptic_matrix.size(0)
        non_spiked_neurons = torch.ones(size, dtype=torch.bool)
        non_spiked_neurons[spiked_neurons] = False
        # recovery
        self.synaptic_matrix[non_spiked_neurons] = 1.0
        return self.synaptic_matrix
    
    def __repr__(self):
        print(f"decay_rate: {self.decay_rate}")
        return str(self.synaptic_matrix)
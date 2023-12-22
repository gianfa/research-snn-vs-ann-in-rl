"""
"""
import torch

class FIFO_buffer():

    def __init__(self, capacity: int = 10):
        assert capacity > 0
        self.capacity = capacity
        self.buffer_ = []

    def __len__(self):
        return len(self.buffer_)
    
    def __repr__(self) -> str:
        return self.buffer_.__str__()

    def push(self, x: object):
        self.buffer_.append(x)
        if len(self.buffer_) >= self.capacity + 1:
            self.buffer_.pop(0)
        return self.buffer_
    

def quantize_to_bins(input_tensor, num_bins):
    """Quantize to bins

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to quantize
    num_bins : int
        num og bins to use for quantization

    Returns
    -------
    torch.Tensor

    Examples
    --------
    >>> random_tensor = torch.rand(3, 3)
    >>> num_bins = 3
    >>> quantized_tensor = quantize_to_bins(random_tensor, num_bins)
    >>> print(random_tensor)
    >>> print(quantized_tensor)
    tensor([[0.3094, 0.7879, 0.9206],
        [0.5843, 0.0456, 0.0780],
        [0.1617, 0.5618, 0.1568]])

    tensor([[0.5000, 1.0000, 1.0000],
            [0.5000, 0.0000, 0.0000],
            [0.0000, 0.5000, 0.0000]])
    """
    bin_width = 1.0 / (num_bins - 1)
    return torch.round(input_tensor / bin_width) * bin_width
    
    
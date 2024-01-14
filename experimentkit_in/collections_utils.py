""""""

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
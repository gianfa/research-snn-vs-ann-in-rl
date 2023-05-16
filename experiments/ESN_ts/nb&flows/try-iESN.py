# %%
from typing import Tuple
import numpy as np

class EchoStateGC_iESN:
    def __init__(
            self,
            Nr: Tuple[int, int], seqDim:int,
            rho: float = 0.9,
            Wr = None, Win = None,
            alpha: float = None, connectivity: float = 1,
            inputScaling: float = None,
            biasScaling: float = 0,
            lambda_val: float = 1,
            nonlinearfunction: str = 'tanh',
            orthonormalWeights = None,
            readout_training = 1,
            ):

        if seqDim != len(Nr):
            raise ValueError('seqDim mismatch')

        self.seqDim = seqDim
        self.Nr = Nr
        self.Ntot = sum(Nr)
        self.rho = rho or 0.9  # spectralRadius
        self.nonlinearfunction = nonlinearfunction or 'tanh'
        self.readout_training = readout_training or 'ridgeregression'
        
        self.connectivity = connectivity or 1
        self.orthonormalWeights = orthonormalWeights or 1
        
        self.Wr = Wr
        self.wrexternallydefined = self.Wr is not None
        self.Win = Win
        self.winexternallydefined = self.Win is not None
        
        self.alpha = alpha or 1  # leakRate
        self.inputScaling = inputScaling or 1
        self.biasScaling = biasScaling or 0.
        self.lambda_val = lambda_val or 1  # regularization

        if not self.winexternallydefined:
            Win = np.array([])
            for i in range(seqDim):
                wtemp = self.inputScaling * (np.random.rand(Nr[i], 1) * 2 - 1)
                Win = np.block([
                    [Win, np.zeros((Win.shape[0], wtemp.shape[1]))],
                    [wtemp, np.zeros((wtemp.shape[0], Win.shape[1]), dtype=wtemp.dtype)]])
            self.Win = Win
        self.Wb = self.biasScaling * (np.random.rand(self.Ntot, 1) * 2 - 1)

        if not self.wrexternallydefined:
            if self.orthonormalWeights:
                Wr = []
                for i in range(seqDim):
                    wwtemp = np.random.rand(Nr[i], Nr[i])


esn = EchoStateGC_iESN(Nr = (5, 5), seqDim = 2)
# %%
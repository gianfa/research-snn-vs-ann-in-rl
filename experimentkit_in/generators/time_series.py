from itertools import combinations
from typing import List, Tuple

import numpy as np

def ts_add_onset(
    onset: float, signal: np.ndarray, baseline: float = 0) -> np.ndarray:
    if onset > 0:
        return np.concatenate((
            np.ones(onset) * baseline,
            signal[signal.shape[0] - onset:]
        ))
    return signal

def ts_generate_constant(
    length: int, baseline: float, onset: float = 0) -> np.ndarray:
    sig = np.ones(length) * baseline
    if onset > 0:
        return ts_add_onset(onset, sig, baseline)
    return sig


def ts_generate_periodic_peak(
    length: int, max: float, step: int, baseline: float = 0, onset: float = 0
) -> np.ndarray:
    sig = np.ones(length) * baseline
    sig[np.arange(0, length, step)] = max
    if onset > 0:
        return ts_add_onset(onset, sig, baseline)
    return sig


def ts_generate_periodic_sin(
    length: int, amplitude: float, phase: float = 1,
    baseline: float = 0, onset: float = 0
) -> np.ndarray:
    sig = baseline + np.sin(np.arange(length) * phase) * amplitude
    if onset > 0:
        return ts_add_onset(onset, sig, baseline=0)
    return sig



# Dataset generators

def tsds_generate_periodic_sin_as_prod_from_params(
    nexamples: int,
    length: int,
    phases: List[float],
    amplitudes: List[float],
    baselines: List[float],
    ncats_to_combine: int = 2,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(phases) == len(amplitudes) == len(baselines)
    ncats = len(phases)
    assert ncats_to_combine <= ncats

    # generate the primitive signals
    #     signal_i x length
    signals = np.array(
        [ts_generate_periodic_sin(
            length, baseline=baselines[i],
            amplitude=amplitudes[i], phase=phases[i]) for i in range(ncats)])

    # generate the signals combinations
    min_combs = np.array(list(combinations(range(ncats), ncats_to_combine)))
    experiment_combs = min_combs[np.random.randint(0, len(min_combs), nexamples)]
    examples = np.array([
        signals[eci, :].prod(axis=axis) for eci in experiment_combs])
    assert examples.shape == (nexamples, length)
    return examples, experiment_combs


def tsds_generate_periodic_sin_as_sum_from_params(
    nexamples: int,
    length: int,
    phases: List[float],
    amplitudes: List[float],
    baselines: List[float],
    ncats_to_combine: int = 3,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(phases) == len(amplitudes) == len(baselines)
    ncats = len(phases)
    assert ncats_to_combine <= ncats

    # generate the primitive signals
    #     signal_i x length
    signals = np.array(
        [ts_generate_periodic_sin(
            length, baseline=baselines[i],
            amplitude=amplitudes[i], phase=phases[i]) for i in range(ncats)])

    # generate the signals combinations
    min_combs = np.array(list(combinations(range(ncats), ncats_to_combine)))
    experiment_combs = min_combs[np.random.randint(0, len(min_combs), nexamples)]
    examples = np.array([
        signals[eci, :].sum(axis=axis) for eci in experiment_combs])
    assert examples.shape == (nexamples, length)
    return examples, experiment_combs


def tsds_generate_periodic_sin_as_sum(
    nexamples: int,
    length: int,
    ncats: int,
    max_phase: float = 2,
    max_amplitude: float = 20,
    max_baselines: float = 0,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:

    phases = np.random.rand(ncats) * max_phase
    amplitudes = np.random.rand(ncats) * max_amplitude
    baselines = np.random.rand(ncats) * max_baselines

    return tsds_generate_periodic_sin_as_sum_from_params(
        nexamples=nexamples,
        length=length,
        phases=phases,
        amplitudes=amplitudes,
        baselines=baselines,
        axis=axis
    )


# Dynamics

def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.

    Credits
    ---------
    `<https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html>`_

    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


def gen_lorenz(
        dt: float = 0.01,
        n_steps: int = 10000,
        xyz_0: tuple = (0., 1., 1.05),
        s=10, r=28, b=2.667):
    """Generate lorenz attractor time series

    Parameters
    ----------
    dt : float, optional
        dt, by default 0.01
    n_steps : int, optional
        Number of time steps, by default 10000
    xyz_0 : tuple, optional
        Initial Conditions, by default (0., 1., 1.05)
    s : int, optional
        s parameter, by default 10
    r : int, optional
        r parameter, by default 28
    b : float, optional
        b parameter, by default 2.667

    Returns
    -------
    np.array
        x-y-z time series, shaped as (num_steps +1) x 3
    """
    xyzs = np.empty((n_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = xyz_0  # Set initial values
    # Step through "time", calculating the
    #   partial derivatives at the current point
    #   and using them to estimate the next point
    for i in range(n_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i], s=s, r=r, b=b) * dt
    return xyzs
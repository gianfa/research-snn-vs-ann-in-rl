# %%
from itertools import combinations
from scipy import integrate
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


def gen_simple_sin(
        length: int,
        frequency: float,
        amplitude: float = 1,
        base_line: float = 0,
        sampling_rate: int = 10) -> np.array:
    t = np.arange(0, length, 1/sampling_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t/length) + base_line

def gen_random_artifacts(
        num_signals: int,
        min_length: int,
        max_length: int,
        min_frequency: float,
        max_frequency: float,
        amplitude: float = 1,
        sampling_rate: int = 100) -> list:
    artifacts = []
    for _ in range(num_signals):
        length = np.random.randint(min_length, max_length + 1)
        frequency = np.random.uniform(min_frequency, max_frequency)
        signal = gen_simple_sin(
            length=length, frequency=frequency,
            amplitude=amplitude, sampling_rate=sampling_rate)
        artifacts.append(signal)
    return artifacts


def add_artifacts_to_signal(signal, artifacts) -> tuple:
    dirty_signal = np.copy(signal)
    labels = np.zeros_like(dirty_signal)
    for artifact in artifacts:
        idx = np.random.randint(len(signal) - len(artifact) + 1)
        dirty_signal[idx:idx+len(artifact)] += artifact
        labels[idx:idx+len(artifact)] = 1
    return dirty_signal, labels


def ts_gen_signal_shifts(
    sig: np.ndarray,
    window_size: int = 5,
    dim: int = 0,
    max_shifts: int = None
) -> np.ndarray:

    max_shifts_ = sig.shape[dim] - window_size + 1
    if not max_shifts or max_shifts > max_shifts_:
        max_shifts = max_shifts_
    shifted_sig = [
        np.roll(sig, shift=-i, axis=dim)[:window_size]
        for i in range(max_shifts)]

    return np.stack(shifted_sig)


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

    Example
    -------
    xyzs = gen_lorenz(s=12, r=30, b=2.700)
    # Plot
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()
    """
    xyzs = np.empty((n_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = xyz_0  # Set initial values
    # Step through "time", calculating the
    #   partial derivatives at the current point
    #   and using them to estimate the next point
    for i in range(n_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i], s=s, r=r, b=b) * dt
    return xyzs

# %%

def duffing(
        t, X,
        alpha: float = 1,
        beta: float = -1,
        gamma: float = 0.5,
        delta: float = 0.3,
        omega: float = 1.0,) -> List:
    """ Derivatives of the duffing oscillator
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> t_eval=np.linspace(t_span[0], t_span[1], n_steps)
    >>> sol = integrate.solve_ivp(
        duffing,
        t_span, X0, t_eval=t_eval,
        args = (alpha, beta, gamma, delta, omega))
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(t_eval, x, label='Posizione')
    >>> plt.plot(t_eval, v, label='Velocità')
    >>> plt.xlabel('Tempo')
    >>> plt.ylabel('Valori')
    >>> plt.legend()
    >>> plt.title('Duffing Oscillator')
    >>> plt.grid(True)
    >>> plt.show()
    """
    x, v = X
    dxdt = v
    dvdt = -delta * v - beta * x - alpha * x**3 + gamma * np.cos(omega * t)
    return [dxdt, dvdt]


def gen_duffing(
    X0: List = [0.5, 0.0],
    t_span: List = (0, 200),
    n_steps: int = 1000,
    alpha: float = 1,
    beta: float = -1,
    gamma: float = 0.5,
    delta: float = 0.3,
    omega: float = 1.0,
) -> np.array:
    """Generate Duffing oscillator
    
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> x, v = gen_duffing()
    >>> plt.figure(figsize=(6, 6))
    >>> plt.plot(x, v, label='Phase plane')
    >>> plt.xlabel('x')
    >>> plt.ylabel('v')
    >>> plt.title('Phase plane')
    >>> plt.grid(True)
    """
    sol = integrate.solve_ivp(
        duffing,
        t_span, X0, t_eval=np.linspace(t_span[0], t_span[1], n_steps),
        args=(alpha, beta, gamma, delta, omega))

    return sol.y.T


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
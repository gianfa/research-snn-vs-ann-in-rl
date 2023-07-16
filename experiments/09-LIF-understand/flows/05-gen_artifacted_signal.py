"""
An example of signal generation with artifacts is shown here.

"""
# %%
import numpy as np
import matplotlib.pyplot as plt


def gen_simple_sin(length: int, frequency: float) -> np.array:
    t = np.arange(length)
    return np.sin(2 * np.pi * frequency * t / length)

def gen_random_artifacts(
        num_signals: int,
        min_length: int,
        max_length: int,
        min_frequency: float,
        max_frequency: float) -> list:
    artifacts = []
    for _ in range(num_signals):
        length = np.random.randint(min_length, max_length + 1)
        frequency = np.random.uniform(min_frequency, max_frequency)
        signal = gen_simple_sin(length, frequency)
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


signal_length = 1000
num_artifacts = 5
min_artifact_length = 5
max_artifact_length = 20
min_artifact_frequency = 1
max_artifact_frequency = 10

main_signal = gen_simple_sin(signal_length, 5)

artifacts = gen_random_artifacts(
    num_artifacts, min_artifact_length, max_artifact_length,
    min_artifact_frequency, max_artifact_frequency)

dirty_signal, labels = add_artifacts_to_signal(main_signal, artifacts)



print("Segnale principale:")
print(main_signal)
print("Artefatti:")
for artifact in artifacts:
    print(artifact)
print("Segnale sporco:")
print(dirty_signal)
print("Etichette:")
print(labels)

fig, axs = plt.subplots(3, 1)
axs[0].plot(main_signal)
axs[0].set_title('main_signal')
axs[1].plot(dirty_signal)
axs[1].set_title('dirty_signal')
axs[2].plot(labels)
axs[2].set_title('labels indices')
fig.tight_layout()
# %%

import matplotlib.pyplot as plt
import numpy as np
import torch

def compare_sample_from_data(
    data: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_samples: int = 3
) -> np.ndarray:
    sampled_idxs = torch.randint(len(data), (1, n_samples)).flatten()

    fig, axs = plt.subplots(len(sampled_idxs), 3, figsize=(4, 6))
    for i, idx in enumerate(sampled_idxs):
        img = data[idx]
        axs[i, 0].imshow(img.squeeze())
        axs[i, 1].text(
            0.5, 0.5, str(y_true[idx].item()), horizontalalignment='center',
        verticalalignment='center', fontsize=25)
        axs[i, 1].axis('off')
        axs[i, 2].text(
            0.5, 0.5, str(y_pred[idx].item()), horizontalalignment='center',
        verticalalignment='center', fontsize=25)
        axs[i, 2].axis('off')
    return axs
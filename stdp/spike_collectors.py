""" Spike Collectors

Here the rules of collecting spikes from raster plot pairs are defined
"""
import torch


# # Two-neurons #

def all_to_all(
    pre_raster: torch.Tensor,
    post_raster: torch.Tensor,
) -> torch.Tensor:
    """Collect spikes according to all pre spikes against all post spikes.

    Parameters
    ----------
    pre_raster: torch.Tensor
        The presynaptic neuron raster
    post_raster: torch.Tensor
        The postsynaptic neuron raster

    Returns
    -------
    torch.Tensor
        A list of tuples containing the pre-post spike indexes pairs.
        [(PRE_IDX, POST_IDX)]
    """

    tpre_tpost = []

    if not pre_raster.any() or not post_raster.any():
        return tpre_tpost

    pre_idxs = pre_raster.argwhere().flatten()
    post_idxs = post_raster.argwhere().flatten()
    idxs_pairs = torch.cartesian_prod(pre_idxs, post_idxs)
    non_same_idxs_pairs = (idxs_pairs[:, 0] - idxs_pairs[:, 1]).argwhere()
    idxs_pairs = idxs_pairs[non_same_idxs_pairs].squeeze()
    assert idxs_pairs.ndim <= 2
    assert idxs_pairs.shape[-1] == 2

    if idxs_pairs.ndim == 2:
        tpre_tpost = [(x[0], x[1]) for x in idxs_pairs.tolist()]
    else:
        tpre_tpost = [tuple(idxs_pairs.squeeze().tolist())]
    return tpre_tpost


def nearest_pre_post_pair(
    pre_raster: torch.Tensor,
    post_raster: torch.Tensor,
) -> torch.Tensor:
    """ Collect spikes as the nearest pre and post pair

    Description
    -----------
    Focused on post synaptic, if no post spikes occur ignore.
    For each post spike:
        choose the nearest pre spike
        delete it with all previous ones.

    Example:
        given the following raster from n0 and n1,
            n0: [1, 1, 0, 0, 1, 0, 0, 1]
            n1: [0, 0, 1, 0, 0, 0, 1, 0],
        where n0 is presynaptic and n1 is postsynaptic,
        1. select the first spike in n1,
        2. in n0, take the spike closest to the one from n1
            n0: [1, (1), 0, 0, 1, 0, 0, 1]
            n1: [0, 0, (1), 0, 0, 0, 1, 0]

    Notes
    -----
    it works, but it doesn't know how to choose between pre and post neurons.
    It's just everything the same.

    Parameters
    ----------
    pre_raster: torch.Tensor
        The presynaptic neuron raster
    post_raster: torch.Tensor
        The postsynaptic neuron raster

    Returns
    -------
    torch.Tensor
        A list of tuples containing the pre-post spike indexes pairs.
        [(PRE_IDX, POST_IDX)]

    Returns
    -------
    torch.Tensor[Tuple[int]]:
        spike index pairs from pre to post as (PRE_IDX, POST_IDX)
    """
    EXCLUSION_VALUE = 1e4  # a trick to exclude previous pre spks in diffs.
    tpre_tpost = []

    if not pre_raster.any() or not post_raster.any():
        return tpre_tpost

    prev_pre_spk_id = None
    pre_spk_ids = pre_raster.argwhere().flatten()
    post_spk_ids = post_raster.argwhere().flatten()
    for post_spk_id_tns in post_spk_ids:  # for each j-th post spike
        post_spk_id = post_spk_id_tns.item()
        pre_post_diffs = pre_spk_ids - post_spk_id
        pre_post_diffs[pre_post_diffs == 0] = EXCLUSION_VALUE

        if prev_pre_spk_id:
            pre_post_diffs[:prev_pre_spk_id + 1] = -EXCLUSION_VALUE

        elected_diff_id = torch.argmin(pre_post_diffs.abs())  # id of the diff
        if abs(pre_post_diffs[elected_diff_id]) > (EXCLUSION_VALUE - 10):
            continue
        elected_diff = pre_post_diffs[elected_diff_id].item()
        elected_id = post_spk_id + elected_diff

        prev_pre_spk_id = elected_diff_id
        tpre_tpost.append((elected_id, post_spk_id))
    return tpre_tpost

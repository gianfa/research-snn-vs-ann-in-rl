"""

"""
import torch


# # Two-neurons #


def nearest_pre_post_pair(
    pre_raster: torch.Tensor,
    post_raster: torch.Tensor,
    v: bool = False,
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
    it works, but it dowsn't know how to choose between pre and post neurons.
    It's just everything the same.

    """
    EXCLUSION_VALUE = 1e4
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
        breakpoint()
        tpre_tpost.append((elected_id, post_spk_id))
    return tpre_tpost

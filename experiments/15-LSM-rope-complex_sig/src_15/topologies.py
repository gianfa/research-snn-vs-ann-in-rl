"""

"""
# %%
import torch


def gen_chain(n: int, m: int, mutual: bool = False):
    adj = torch.zeros(n, m)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        if mutual:
            adj[i + 1, i] = 1
    return adj


def gen_candidates_for_rope(n: int, m: int, radius: int = 1, degree: int = 2):
    """"chain-like, with links in a neighborhood of the diagonal"""
    if radius < 0:
        raise ValueError("radius must be positive")
    if radius * 2 + 1 > m:
        raise ValueError("radius too big: must be <= (m-1)/2")
    if degree > radius * 2:
        raise ValueError()
    if degree > m - radius:
        raise ValueError(
            f"degree too big for the given radius: try with {m - radius}")

    adj = torch.zeros(n, m)
    all_candidates = []
    for i in range(n):
        candidates = torch.arange(i - radius - 1, i + radius + 2)
        candidates = candidates[
            (candidates >= 0)
            & (candidates < m)
            & (candidates != i)]
        all_candidates.append(candidates)
    return all_candidates

def gen_rope(n: int, m: int, radius: int = 1, degree: int = 2):
    """"chain-like, with links in a neighborhood of the diagonal"""
    if radius < 0:
        raise ValueError("radius must be positive")
    if radius * 2 + 1 > m:
        raise ValueError("radius too big: must be <= (m-1)/2")
    if degree > radius * 2:
        raise ValueError()
    if degree > m - radius:
        raise ValueError(
            f"degree too big for the given radius: try with {m - radius}")

    adj = torch.zeros(n, m)
    for i in range(n):
        candidates = torch.arange(i - radius - 1, i + radius + 2)
        candidates = candidates[
            (candidates >= 0)
            & (candidates < m)
            & (candidates != i)]
        for _ in range(degree):
            pos = candidates[torch.randperm(len(candidates))[0]].item()
            adj[i, pos] = 1
            candidates = candidates[candidates != pos]
    return adj


def gen_by_connection_degree(n: int, m: int, degree: int):
    adj = torch.zeros(size=(n, m))
    for i in range(adj.shape[0]):
        candidates = None
        while candidates is None:
            candidates = torch.randperm(m)[:degree]
            if i in candidates:
                candidates = None
        adj[i, candidates] = 1
    return adj

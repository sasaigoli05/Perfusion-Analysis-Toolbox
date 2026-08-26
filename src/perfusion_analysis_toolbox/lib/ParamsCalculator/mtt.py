import numpy as np


def cal(cbv, cbf, mask):
    """MTT = CBV / CBF by the central volume theorem, zero outside the mask."""
    mtt = np.zeros_like(cbv, dtype=float)
    valid = (mask > 0) & (cbf > 1e-6)
    mtt[valid] = cbv[valid] / cbf[valid]
    return mtt

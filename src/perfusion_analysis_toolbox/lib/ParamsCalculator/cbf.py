from .svd import DSC_SVD


def cal(conc, aif, mask, svd_frac=0.20):
    """
    Deconvolve the concentration series against the AIF to get CBF and Tmax.

    The AIF dict carries the TR it was fitted at; forward it so the voxel
    time base matches the AIF time base. Without this DSC_SVD falls back to
    its own default TR and the two grids drift apart.

    Returns
    -------
    cbf  : (Z, Y, X) peak of the residue function
    tmax : (Z, Y, X) time-to-peak of the residue function, in seconds
    """
    kwargs = {}
    if isinstance(aif, dict) and aif.get("TR"):
        kwargs["TR"] = float(aif["TR"])
    return DSC_SVD(conc, aif, mask, svd_frac=svd_frac, **kwargs)

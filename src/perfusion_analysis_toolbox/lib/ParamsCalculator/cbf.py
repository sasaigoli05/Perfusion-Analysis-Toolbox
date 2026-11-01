from .svd import cal as deconvolve


def cal(conc, aif, mask, svd_frac=0.20, method="svd", max_delay_s=16.0,
        extras=None, bayes_kwargs=None):
    """
    Deconvolve the concentration series against the AIF to get CBF and Tmax.

    The AIF dict carries the TR it was fitted at; forward it so the voxel
    time base matches the AIF time base. Without this the deconvolution
    falls back to its own default TR and the two grids drift apart.

    Parameters
    ----------
    method : 'svd' for the Toeplitz form (default), or 'osvd' for
        block-circulant, delay-insensitive deconvolution. See svd.cal on
        why oSVD is not yet the default.
    max_delay_s : oSVD only; longest residue-peak delay treated as real.

    Returns
    -------
    cbf  : (Z, Y, X) peak of the residue function
    tmax : (Z, Y, X) time-to-peak of the residue function, in seconds
    """
    kwargs = {}
    if isinstance(aif, dict) and aif.get("TR"):
        kwargs["default_TR"] = float(aif["TR"])
    return deconvolve(
        conc,
        aif,
        mask,
        svd_frac=svd_frac,
        method=method,
        max_delay_s=max_delay_s,
        extras=extras,
        bayes_kwargs=bayes_kwargs,
        **kwargs,
    )

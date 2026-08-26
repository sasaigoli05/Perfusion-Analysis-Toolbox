import numpy as np
from scipy.linalg import toeplitz
from scipy.ndimage import zoom


def _resample_mask_to_conc(mask, conc_spatial_shape):
    """
    Nearest-neighbor resample of a (Z,Y,X) mask to match conc spatial shape.
    Works for any integer sizes; keeps it binary.
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    srcZ, srcY, srcX = mask.shape
    tgtZ, tgtY, tgtX = conc_spatial_shape

    # Compute zoom factors
    zf = (tgtZ / float(srcZ)) if srcZ != tgtZ else 1.0
    yf = (tgtY / float(srcY)) if srcY != tgtY else 1.0
    xf = (tgtX / float(srcX)) if srcX != tgtX else 1.0

    if (zf, yf, xf) != (1.0, 1.0, 1.0):
        # order=0 → nearest neighbor (preserve labels)
        mask_rs = zoom(mask, zoom=(zf, yf, xf), order=0, mode="nearest")
        mask_rs = (mask_rs > 0.5).astype(bool)
    else:
        mask_rs = mask > 0

    return mask_rs


def _resample_to_time(y, t_src, t_dst):
    """1D linear resample curve y(t_src) → y(t_dst). Robust fallbacks."""
    y = np.asarray(y, float).reshape(-1)
    t_src = np.asarray(t_src, float).reshape(-1)
    t_dst = np.asarray(t_dst, float).reshape(-1)

    if (
        t_src.size < 2
        or not np.all(np.isfinite(t_src))
        or not np.all(np.isfinite(y))
    ):
        # Fallback: truncate or pad with zeros
        out = np.zeros_like(t_dst, float)
        n = min(out.size, y.size)
        out[:n] = y[:n]
        return out

    # Clamp ends with edge values
    return np.interp(t_dst, t_src, y, left=y[0], right=y[-1])


def _build_G(aif_vec):
    """Toeplitz convolution matrix for AIF."""
    aif_vec = np.asarray(aif_vec, float).reshape(-1)
    first_col = aif_vec
    first_row = np.zeros_like(aif_vec)
    first_row[0] = aif_vec[0]
    return toeplitz(first_col, first_row)


def DSC_SVD(
    conc, aif_curve, mask, TR=1.55, aif_time=None, svd_frac=0.20, verbose=True
):
    """
    Truncated-SVD deconvolution for DSC perfusion with automatic mask resizing.

    Parameters
    ----------
    conc : (Z,Y,X,T) float array (concentration series)
    aif_curve : (T_aif,) array OR dict with key 'gv'
    mask : (Z,Y,X) bool/uint8 (any size; will be resampled to conc spatial shape)
    TR : seconds per frame (used if aif_time is absent)
    aif_time : optional (T_aif,) time vector for the AIF
    svd_frac : threshold as fraction of max singular value
    verbose : print reconciliation info

    Returns
    -------
    cbf_map  : (Z,Y,X)
    tmax_map : (Z,Y,X) argmax index of residue
    """
    conc = np.asarray(conc, float)
    if conc.ndim != 4:
        raise ValueError(f"conc must be (Z,Y,X,T), got {conc.shape}")
    nZ, nY, nX, nT = conc.shape

    # Resample mask → conc spatial shape
    mask_rs = _resample_mask_to_conc(mask, (nZ, nY, nX))
    if verbose:
        print(
            f"[SVD] conc shape: {conc.shape}, mask (in) shape: {np.asarray(mask).shape}, mask (rs) shape: {mask_rs.shape}"
        )

    # Prepare AIF
    if isinstance(aif_curve, dict):
        aif_vec = np.asarray(aif_curve.get("gv", None), float).reshape(-1)
        aif_time_vec = (
            np.asarray(aif_curve.get("time", None), float).reshape(-1)
            if ("time" in aif_curve)
            else None
        )
    else:
        aif_vec = np.asarray(aif_curve, float).reshape(-1)
        aif_time_vec = None

    # Target time base for voxels
    t_vox = np.arange(nT, dtype=float) * TR

    # Resample AIF to voxel time base if needed
    if aif_time is not None:
        aif_use = _resample_to_time(aif_vec, aif_time, t_vox)
    elif aif_time_vec is not None and aif_time_vec.size == aif_vec.size:
        aif_use = _resample_to_time(aif_vec, aif_time_vec, t_vox)
    else:
        # Assume uniform but maybe different length
        if aif_vec.size == nT:
            aif_use = aif_vec
        else:
            aif_use = _resample_to_time(
                aif_vec, np.linspace(0, 1, aif_vec.size), np.linspace(0, 1, nT)
            )

    # Mild smoothing of AIF (3-point)
    if nT >= 3:
        aif_use = aif_use.copy()
        aif_use[1:-1] = (
            aif_use[:-2] + 4.0 * aif_use[1:-1] + aif_use[2:]
        ) / 6.0

    # Build convolution matrix and its truncated inverse
    G = _build_G(aif_use)  # (T,T)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    tau = float(svd_frac) * float(np.max(S)) if S.size else 0.0
    S_inv = np.where(S >= max(tau, 1e-12), 1.0 / S, 0.0)
    Ginv = (Vt.T * S_inv) @ U.T  # (T,T)

    cbf_map = np.zeros((nZ, nY, nX), float)
    tmax_map = np.zeros((nZ, nY, nX), int)

    # Compute only within mask (slice-wise batching)
    for z in range(nZ):
        msl = mask_rs[z]
        if not msl.any():
            continue
        rr, cc = np.where(msl)

        # Gather curves at (z, rr, cc)
        C_T = conc[z, rr, cc, :].T  # (T, N)
        R = (1.0 / TR) * (Ginv @ C_T)  # (T, N)
        R = R.T  # (N, T)

        cbf_map[z, rr, cc] = np.max(np.abs(R), axis=1)
        tmax_map[z, rr, cc] = np.argmax(R, axis=1)

    return cbf_map, tmax_map


def cal(conc, aif_dict, mask, default_TR=1.55, svd_frac=0.20, verbose=True):
    """
    Wrapper for pipeline use.
      conc     : (Z,Y,X,T)
      aif_dict : dict with keys 'gv' and optionally 'time', or array
      mask     : (Z,Y,X) (any size; will be resampled)
    """
    if isinstance(aif_dict, dict):
        aif_curve = aif_dict.get("gv", None)
        aif_time = aif_dict.get("time", None)
        if aif_curve is None:
            raise ValueError("aif_dict missing 'gv' curve")
    else:
        aif_curve = np.asarray(aif_dict, float).ravel()
        aif_time = None

    # If time is provided, estimate TR from it; else use default
    if aif_time is not None and np.asarray(aif_time).size >= 2:
        dt = float(np.median(np.diff(np.asarray(aif_time, float))))
        TR = dt if dt > 0 else float(default_TR)
    else:
        TR = float(default_TR)

    cbf, tmax = DSC_SVD(
        conc,
        aif_curve,
        mask,
        TR=TR,
        aif_time=aif_time,
        svd_frac=svd_frac,
        verbose=verbose,
    )
    return cbf, tmax

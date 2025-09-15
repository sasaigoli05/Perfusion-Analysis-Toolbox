import numpy as np
from scipy.linalg import toeplitz


def DSC_SVD(conc, aif_curve, mask, TR=1.55, aif_time=None, svd_frac=0.20):
    """
    Truncated-SVD deconvolution for DSC perfusion.

    Parameters
    ----------
    conc : (Z,R,C,T) float array (concentration series)
    aif_curve : (T_aif,) array or dict-like with key 'gv'
    mask : (Z,R,C) bool/uint8
    TR : seconds per frame (if aif_time missing)
    aif_time : optional (T_aif,) time vector for the AIF
    svd_frac : threshold as fraction of max singular value

    Returns
    -------
    cbf_map  : (Z,R,C)
    tmax_map : (Z,R,C) argmax index of residue
    """
    conc = np.asarray(conc, float)
    mask = np.asarray(mask, bool)
    if conc.ndim != 4:
        raise ValueError(f"conc must be (Z,R,C,T), got {conc.shape}")
    if mask.shape != conc.shape[:3]:
        raise ValueError(f"mask shape {mask.shape} != conc spatial shape {conc.shape[:3]}")
    nS, nR, nC, nT = conc.shape

    # --- local helpers as closures ---
    def _resample_to_time(y, t_src, t_dst):
        y = np.asarray(y, float).reshape(-1)
        t_src = np.asarray(t_src, float).reshape(-1)
        t_dst = np.asarray(t_dst, float).reshape(-1)
        if t_src.size < 2 or not np.all(np.isfinite(t_src)) or not np.all(np.isfinite(y)):
            if t_dst.size <= y.size:
                return y[: t_dst.size]
            out = np.zeros_like(t_dst); out[: y.size] = y; return out
        if t_src.size == 2:
            return np.interp(t_dst, (t_src[0], t_src[-1]), (y[0], y[-1]), left=y[0], right=y[-1])
        return np.interp(t_dst, t_src, y)

    def _build_G(aif_vec):
        first_col = aif_vec
        first_row = np.zeros_like(aif_vec); first_row[0] = aif_vec[0]
        return toeplitz(first_col, first_row)

    # time bases
    t_vox = np.arange(0.0, nT * TR, TR, dtype=float)
    aif_vec = np.asarray(aif_curve["gv"] if isinstance(aif_curve, dict) else aif_curve, float).reshape(-1)

    if aif_time is not None:
        aif_use = _resample_to_time(aif_vec, np.asarray(aif_time, float).reshape(-1), t_vox)
    else:
        aif_use = aif_vec if aif_vec.size == nT else _resample_to_time(aif_vec, np.arange(aif_vec.size), np.linspace(0, aif_vec.size - 1, num=nT))

    # light smoothing (3-point)
    if nT >= 3:
        aif_use[1:-1] = (aif_use[:-2] + 4.0 * aif_use[1:-1] + aif_use[2:]) / 6.0

    G = _build_G(aif_use)
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    tau = float(svd_frac) * float(np.max(S)) if S.size else 0.0
    S_inv = np.where(S >= max(tau, 1e-12), 1.0 / S, 0.0)
    Ginv = (Vt.T * S_inv) @ U.T  # (nT x nT)

    cbf_map  = np.zeros((nS, nR, nC), float)
    tmax_map = np.zeros((nS, nR, nC), int)

    for s in range(nS):
        msl = mask[s]
        if not msl.any():
            continue
        rr, cc = np.where(msl)
        C_T = conc[s, rr, cc, :].T          # (T, N)
        R = (1.0 / TR) * (Ginv @ C_T)       # (T, N)
        R = R.T                              # (N, T)
        cbf_map[s, rr, cc]  = np.max(np.abs(R), axis=1)
        tmax_map[s, rr, cc] = np.argmax(R, axis=1)

    return cbf_map, tmax_map


def cal(conc, aif_dict, mask, default_TR=1.55, svd_frac=0.20):
    """
    Pipeline wrapper.
    conc     : (Z,R,C,T)
    aif_dict : expects keys 'gv' and optionally 'time'
    mask     : (Z,R,C)
    """
    if isinstance(aif_dict, dict):
        aif_curve = aif_dict.get("gv", None)
        aif_time  = aif_dict.get("time", None)
        if aif_curve is None:
            raise ValueError("aif_dict missing 'gv' curve")
    else:
        aif_curve = np.asarray(aif_dict, float).ravel()
        aif_time  = None

    TR = float(np.median(np.diff(aif_time))) if (isinstance(aif_time, np.ndarray) and aif_time.size >= 2) else float(default_TR)
    cbf, tmax = DSC_SVD(conc, aif_curve, mask, TR=TR, aif_time=aif_time, svd_frac=svd_frac)
    return cbf, tmax

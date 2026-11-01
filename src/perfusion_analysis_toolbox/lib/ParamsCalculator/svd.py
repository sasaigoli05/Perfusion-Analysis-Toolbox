import numpy as np
from scipy.linalg import circulant, toeplitz
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


def _build_circulant(aif_vec, shift):
    """
    Circulant convolution matrix for a bolus delay of `shift` frames.

    Rolling the AIF forward delays its arrival, so deconvolving against it
    recovers the residue function of tissue that fills `shift` frames late.
    Unlike the Toeplitz form, the circulant matrix wraps, which keeps the
    late tail of the residue from folding back as a spurious early peak.
    """
    aif_vec = np.asarray(aif_vec, float).reshape(-1)
    return circulant(np.roll(aif_vec, int(shift)))


def _pseudoinverse(G, svd_frac):
    """Truncated-SVD pseudoinverse, cutting at svd_frac * max singular value."""
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    tau = float(svd_frac) * float(np.max(S)) if S.size else 0.0
    S_inv = np.where(S >= max(tau, 1e-12), 1.0 / S, 0.0)
    return (Vt.T * S_inv) @ U.T


def _prepare_aif(aif_curve, aif_time, nT, TR):
    """
    Return the AIF resampled onto the voxel time base and lightly smoothed.

    Accepts either a bare curve or the AIF dict the pipeline passes around
    (keys 'gv' and, optionally, 'time').
    """
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

    t_vox = np.arange(nT, dtype=float) * TR

    if aif_time is not None:
        aif_use = _resample_to_time(aif_vec, aif_time, t_vox)
    elif aif_time_vec is not None and aif_time_vec.size == aif_vec.size:
        aif_use = _resample_to_time(aif_vec, aif_time_vec, t_vox)
    elif aif_vec.size == nT:
        aif_use = aif_vec
    else:
        aif_use = _resample_to_time(
            aif_vec, np.linspace(0, 1, aif_vec.size), np.linspace(0, 1, nT)
        )

    # Mild 3-point smoothing, matching the temporal resolution of the
    # tissue curves the AIF is deconvolved out of.
    aif_use = np.asarray(aif_use, float).copy()
    if nT >= 3:
        aif_use[1:-1] = (
            aif_use[:-2] + 4.0 * aif_use[1:-1] + aif_use[2:]
        ) / 6.0
    return aif_use


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
    tmax_map : (Z,Y,X) time-to-peak of the residue function, in seconds
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

    aif_use = _prepare_aif(aif_curve, aif_time, nT, TR)

    # Build convolution matrix and its truncated inverse
    G = _build_G(aif_use)  # (T,T)
    Ginv = _pseudoinverse(G, svd_frac)  # (T,T)

    cbf_map = np.zeros((nZ, nY, nX), float)
    tmax_map = np.zeros((nZ, nY, nX), float)

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

        # The residue function is non-negative by construction; anything
        # below zero is SVD ringing. Taking |R| let a negative lobe set the
        # peak, which inflates CBF most where the true peak is smallest --
        # that is, in the ischemic core. Clip first so CBF and Tmax are both
        # read off the same curve.
        R = np.maximum(R, 0.0)
        cbf_map[z, rr, cc] = np.max(R, axis=1)
        tmax_map[z, rr, cc] = np.argmax(R, axis=1) * TR

    return cbf_map, tmax_map


# Truncation levels tried per voxel by the adaptive method, least
# regularised first. Matches the 5%-95% ladder used by PyPET.
ADAPTIVE_FRACS = np.round(np.arange(0.05, 1.0, 0.05), 2)

# Oscillation-index ceiling. A residue function above this is ringing rather
# than decaying, so the next coarser truncation is tried instead. 0.035 is
# PyPET's default; Wu 2003 uses 0.095-0.15 depending on noise level.
OI_THRESHOLD = 0.035


def _oscillation_index(R, eps=1e-12):
    """
    Wu 2003 oscillation index, vectorised over voxels.

    OI = sum |R[j] - 2 R[j-1] + R[j-2]| / (L * max|R|)

    The second difference is what distinguishes a residue that decays from
    one that rings. It is computed on the *signed* residue: taking the
    absolute value first -- as the reference implementations do -- folds
    negative lobes upward and turns each zero crossing into a cusp, which
    inflates the very quantity being measured. Normalising by max|R| keeps
    the index scale-free.

    R : (N, L) residue functions
    """
    R = np.asarray(R, float)
    if R.shape[1] < 3:
        return np.zeros(R.shape[0])
    curvature = np.abs(np.diff(R, n=2, axis=1)).sum(axis=1)
    scale = np.maximum(np.abs(R).max(axis=1), eps)
    return curvature / (R.shape[1] * scale)


def DSC_oSVD(
    conc,
    aif_curve,
    mask,
    TR=1.5,
    aif_time=None,
    svd_frac=0.20,
    max_delay_s=16.0,
    verbose=True,
):
    """
    Block-circulant (delay-insensitive) SVD deconvolution.

    Ostergaard 1996 / Wu 2003. The Toeplitz form used by :func:`DSC_SVD`
    assumes the bolus reaches every voxel at t=0, so tissue that fills late
    has its residue function truncated and its CBF underestimated -- the
    dominant bias in stroke, where delayed arrival is the thing being
    measured. Deconvolving against a circulant matrix instead lets the
    residue shift freely in time, and zero-padding the series to twice its
    length keeps the wrap-around from folding the late tail back onto the
    early frames.

    Tmax comes out of the same fit: it is the position of the residue peak,
    which under this formulation carries the bolus delay rather than having
    it regressed away.

    Parameters
    ----------
    conc : (Z,Y,X,T) concentration series
    aif_curve : (T_aif,) array, or the pipeline's AIF dict
    mask : (Z,Y,X) brain mask; resampled to conc if needed
    TR : seconds per frame
    aif_time : optional time vector for the AIF
    svd_frac : truncation threshold as a fraction of the largest singular value
    max_delay_s : longest residue-peak delay treated as physiological. Peaks
        beyond this are wrap-around artefact, so the search stops there.

    Returns
    -------
    cbf_map  : (Z,Y,X) peak of the residue function
    tmax_map : (Z,Y,X) time-to-peak of the residue function, in seconds
    """
    conc = np.asarray(conc, float)
    if conc.ndim != 4:
        raise ValueError(f"conc must be (Z,Y,X,T), got {conc.shape}")
    nZ, nY, nX, nT = conc.shape

    mask_rs = _resample_mask_to_conc(mask, (nZ, nY, nX))
    aif_use = _prepare_aif(aif_curve, aif_time, nT, TR)

    # Zero-pad to 2T so the circular convolution does not alias.
    L = 2 * nT
    aif_pad = np.zeros(L, float)
    aif_pad[:nT] = aif_use

    Ginv = _pseudoinverse(_build_circulant(aif_pad, 0), svd_frac)  # (L,L)

    # Residue peaks later than max_delay_s are wrap-around, not physiology.
    n_search = int(round(float(max_delay_s) / TR)) + 1
    n_search = max(1, min(L, n_search))

    if verbose:
        print(
            f"[oSVD] conc {conc.shape}, padded length {L}, "
            f"svd_frac={svd_frac}, peak search 0-{(n_search - 1) * TR:.1f}s"
        )

    cbf_map = np.zeros((nZ, nY, nX), float)
    tmax_map = np.zeros((nZ, nY, nX), float)

    for z in range(nZ):
        msl = mask_rs[z]
        if not msl.any():
            continue
        rr, cc = np.where(msl)

        C = np.zeros((L, len(rr)), float)
        C[:nT, :] = conc[z, rr, cc, :].T

        R = ((1.0 / TR) * (Ginv @ C)).T  # (N, L)

        # 3-point moving average stabilises the peak against ringing.
        if L >= 3:
            R[:, 1:-1] = (R[:, :-2] + R[:, 1:-1] + R[:, 2:]) / 3.0
        R = np.maximum(R[:, :n_search], 0.0)

        cbf_map[z, rr, cc] = np.max(R, axis=1)
        tmax_map[z, rr, cc] = np.argmax(R, axis=1) * TR

    if verbose and mask_rs.any():
        tvals = tmax_map[mask_rs]
        print(
            f"[oSVD] CBF>0 fraction: {(cbf_map[mask_rs] > 0).mean():.3f}  "
            f"Tmax median {np.median(tvals):.2f}s  "
            f"p90 {np.percentile(tvals, 90):.2f}s"
        )

    return cbf_map, tmax_map


def DSC_oSVD_adaptive(
    conc,
    aif_curve,
    mask,
    TR=1.5,
    aif_time=None,
    max_delay_s=16.0,
    oi_threshold=OI_THRESHOLD,
    verbose=True,
):
    """
    Block-circulant deconvolution with per-voxel oscillation-index
    regularisation (Wu 2003; as implemented in PyPET).

    Removes svd_frac as a tuned parameter. Rather than one truncation
    threshold for the whole brain, each voxel takes the least-regularised
    threshold whose residue function is not ringing -- judged by the
    oscillation index. Noisy voxels get more smoothing and clean ones keep
    their temporal detail, which is the right trade in a map whose whole
    purpose is to distinguish low-flow tissue from noise.

    The reference implementation loops in Python over every voxel and every
    threshold, which is intractable at 10^6 voxels. Here the pseudoinverse
    for each threshold is built once from a single SVD, and voxels are
    resolved a whole slice at a time, coarsening only those that have not
    yet met the criterion.

    Parameters
    ----------
    conc : (Z,Y,X,T) concentration series
    aif_curve : (T_aif,) array, or the pipeline's AIF dict
    mask : (Z,Y,X) brain mask
    TR : seconds per frame
    aif_time : optional time vector for the AIF
    max_delay_s : longest residue-peak delay treated as physiological
    oi_threshold : oscillation-index ceiling

    Returns
    -------
    cbf_map  : (Z,Y,X) peak of the residue function
    tmax_map : (Z,Y,X) time-to-peak of the residue function, in seconds
    """
    conc = np.asarray(conc, float)
    if conc.ndim != 4:
        raise ValueError(f"conc must be (Z,Y,X,T), got {conc.shape}")
    nZ, nY, nX, nT = conc.shape

    mask_rs = _resample_mask_to_conc(mask, (nZ, nY, nX))
    aif_use = _prepare_aif(aif_curve, aif_time, nT, TR)

    L = 2 * nT
    aif_pad = np.zeros(L, float)
    aif_pad[:nT] = aif_use

    # One SVD, re-thresholded per level: the factors do not depend on the
    # truncation, only the reciprocal singular values do.
    U, S, Vt = np.linalg.svd(_build_circulant(aif_pad, 0), full_matrices=False)
    s_max = float(np.max(S)) if S.size else 0.0
    ginv_levels = []
    for frac in ADAPTIVE_FRACS:
        s_inv = np.where(S >= max(frac * s_max, 1e-12), 1.0 / S, 0.0)
        ginv_levels.append((Vt.T * s_inv) @ U.T)

    n_search = int(round(float(max_delay_s) / TR)) + 1
    n_search = max(1, min(nT, n_search))

    if verbose:
        print(
            f"[oSVD-OI] conc {conc.shape}, padded length {L}, "
            f"{len(ADAPTIVE_FRACS)} truncation levels, "
            f"OI threshold {oi_threshold}, peak search "
            f"0-{(n_search - 1) * TR:.1f}s"
        )

    cbf_map = np.zeros((nZ, nY, nX), float)
    tmax_map = np.zeros((nZ, nY, nX), float)
    frac_map = np.zeros((nZ, nY, nX), float)

    for z in range(nZ):
        msl = mask_rs[z]
        if not msl.any():
            continue
        rr, cc = np.where(msl)
        n_vox = len(rr)

        C = np.zeros((L, n_vox), float)
        C[:nT, :] = conc[z, rr, cc, :].T

        resolved = np.zeros(n_vox, dtype=bool)
        chosen = np.zeros((n_vox, n_search), float)
        chosen_frac = np.zeros(n_vox, float)

        for fi, frac in enumerate(ADAPTIVE_FRACS):
            pending = ~resolved
            if not pending.any():
                break

            R = ((1.0 / TR) * (ginv_levels[fi] @ C[:, pending])).T
            oi = _oscillation_index(R)
            accept = oi <= oi_threshold

            # Last level: everything still pending has to take it.
            if fi == len(ADAPTIVE_FRACS) - 1:
                accept = np.ones_like(accept)

            if accept.any():
                idx = np.flatnonzero(pending)[accept]
                chosen[idx] = np.maximum(R[accept, :n_search], 0.0)
                chosen_frac[idx] = frac
                resolved[idx] = True

        cbf_map[z, rr, cc] = chosen.max(axis=1)
        tmax_map[z, rr, cc] = chosen.argmax(axis=1) * TR
        frac_map[z, rr, cc] = chosen_frac

    if verbose and mask_rs.any():
        fr = frac_map[mask_rs]
        tv = tmax_map[mask_rs]
        print(
            f"[oSVD-OI] chosen svd_frac: median {np.median(fr):.2f}, "
            f"p10 {np.percentile(fr, 10):.2f}, p90 {np.percentile(fr, 90):.2f}"
        )
        print(
            f"[oSVD-OI] CBF>0 fraction: {(cbf_map[mask_rs] > 0).mean():.3f}  "
            f"Tmax median {np.median(tv):.2f}s  "
            f"p90 {np.percentile(tv, 90):.2f}s"
        )

    return cbf_map, tmax_map


def cal(
    conc,
    aif_dict,
    mask,
    default_TR=1.55,
    svd_frac=0.20,
    method="svd",
    max_delay_s=16.0,
    verbose=True,
    extras=None,
    bayes_kwargs=None,
):
    """
    Wrapper for pipeline use.
      conc        : (Z,Y,X,T)
      aif_dict    : dict with keys 'gv' and optionally 'time', or array
      mask        : (Z,Y,X) (any size; will be resampled)
      method      : 'svd' (Toeplitz; the default), 'osvd_oi'
                    (block-circulant with per-voxel oscillation-index
                    regularisation, which ignores svd_frac), or 'osvd'
                    (block-circulant, delay-insensitive). oSVD scores worse
                    at svd_frac=0.20, which was tuned for the Toeplitz form;
                    re-sweep the threshold before switching the default.
      'bayes'     : MAP parametric deconvolution over a gamma transit-time
                    distribution, regularised by physiological priors and a
                    spatial prior on the shape parameters instead of by
                    smoothing. Ignores svd_frac. See bayes.py.
      max_delay_s : oSVD and bayes; longest residue delay treated as real
      extras      : optional dict, populated with any additional maps the
                    chosen method produces (bayes yields MTT and CTH directly
                    from the fitted transit-time distribution). Left untouched
                    by the SVD methods, which cannot produce them.
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

    method = str(method).lower()
    if method in ("bayes", "bayesian", "map"):
        from .bayes import DSC_bayes

        cbf, tmax, produced = DSC_bayes(
            conc,
            aif_curve,
            mask,
            TR=TR,
            aif_time=aif_time,
            max_delay_s=max_delay_s,
            verbose=verbose,
            **(bayes_kwargs or {}),
        )
        if extras is not None:
            extras.update(produced)
        return cbf, tmax
    if method == "osvd":
        return DSC_oSVD(
            conc,
            aif_curve,
            mask,
            TR=TR,
            aif_time=aif_time,
            svd_frac=svd_frac,
            max_delay_s=max_delay_s,
            verbose=verbose,
        )
    if method == "svd":
        return DSC_SVD(
            conc,
            aif_curve,
            mask,
            TR=TR,
            aif_time=aif_time,
            svd_frac=svd_frac,
            verbose=verbose,
        )
    if method in ("osvd_oi", "oi"):
        return DSC_oSVD_adaptive(
            conc,
            aif_curve,
            mask,
            TR=TR,
            aif_time=aif_time,
            max_delay_s=max_delay_s,
            verbose=verbose,
        )
    raise ValueError(f"Unknown deconvolution method {method!r}")

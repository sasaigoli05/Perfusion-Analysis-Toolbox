import numpy as np
from scipy.linalg import toeplitz, circulant
from scipy.ndimage import zoom


# ==============================================================================
# HELPERS
# ==============================================================================

def _resample_mask_to_conc(mask, conc_spatial_shape):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    srcZ, srcY, srcX = mask.shape
    tgtZ, tgtY, tgtX = conc_spatial_shape
    zf = tgtZ / float(srcZ)
    yf = tgtY / float(srcY)
    xf = tgtX / float(srcX)
    mask_rs = zoom(mask, zoom=(zf, yf, xf), order=0, mode="nearest")
    return mask_rs > 0.5


def _resample_to_time(y, t_src, t_dst):
    return np.interp(t_dst, t_src, y, left=y[0], right=y[-1])


def _build_toeplitz(aif_vec):
    """Standard lower-triangular Toeplitz convolution matrix."""
    return toeplitz(aif_vec, np.r_[aif_vec[0], np.zeros(len(aif_vec) - 1)])


def _build_circulant(aif_vec, shift):
    """
    Circulant convolution matrix for oSVD.
    Shifting the AIF by `shift` frames simulates bolus delay of shift*TR seconds.
    Rolling the AIF forward (np.roll with positive shift) delays it.
    """
    aif_shifted = np.roll(aif_vec, shift)
    return circulant(aif_shifted)


def _pseudoinverse(G, svd_frac):
    """Truncated SVD pseudoinverse with threshold svd_frac * max(S)."""
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    tau = svd_frac * np.max(S)
    S_inv = np.where(S >= tau, 1.0 / S, 0.0)
    return (Vt.T * S_inv) @ U.T


def _svd_factors(G):
    """Return (U, S, Vt) of G without computing Ginv — used by adaptive oSVD."""
    return np.linalg.svd(G, full_matrices=False)


def _apply_threshold(U, S, Vt, svd_frac):
    """Ginv for a given svd_frac, given precomputed U,S,Vt."""
    tau = svd_frac * S[0]
    S_inv = np.where(S >= tau, 1.0 / S, 0.0)
    return (Vt.T * S_inv) @ U.T


def _oscillation_index(R):
    """
    OI = sum(|R[t+1]-R[t]|) / (2*(nT-1)*max(R))  vectorised over voxels.
    R: (nVox, nT), already non-negative.
    Returns OI (nVox,).
    """
    max_R = R.max(axis=1)
    diffs = np.abs(np.diff(R, axis=1)).sum(axis=1)
    return diffs / (2.0 * (R.shape[1] - 1) * np.maximum(max_R, 1e-10))


# ==============================================================================
# STANDARD SVD  (kept for reference / fallback)
# ==============================================================================

def DSC_SVD(
    conc,
    aif_curve,
    mask,
    TR=1.5,
    aif_time=None,
    svd_frac=0.02,
    verbose=True,
):
    """Standard (delay-sensitive) truncated SVD deconvolution."""
    conc = np.asarray(conc, float)
    nZ, nY, nX, nT = conc.shape
    mask_rs = _resample_mask_to_conc(mask, (nZ, nY, nX))

    if isinstance(aif_curve, dict):
        aif_vec      = np.asarray(aif_curve["gv"],   float)
        aif_time_vec = np.asarray(aif_curve["time"], float)
    else:
        aif_vec      = np.asarray(aif_curve, float)
        aif_time_vec = None

    t_vox = np.arange(nT) * TR
    aif_use = (_resample_to_time(aif_vec, aif_time_vec, t_vox)
               if aif_time_vec is not None else aif_vec)

    G    = _build_toeplitz(aif_use)
    Ginv = _pseudoinverse(G, svd_frac)

    cbf_map  = np.zeros((nZ, nY, nX))
    tmax_map = np.zeros((nZ, nY, nX))

    for z in range(nZ):
        rr, cc = np.where(mask_rs[z])
        if len(rr) == 0:
            continue
        C = conc[z, rr, cc, :].T          # (nT, nVox)
        R = (1.0 / TR) * (Ginv @ C)       # (nT, nVox)
        R = R.T                            # (nVox, nT)
        if nT >= 3:
            R[:, 1:-1] = (R[:, :-2] + R[:, 1:-1] + R[:, 2:]) / 3.0
        R = np.maximum(R, 0)
        cbf_map[z, rr, cc]  = np.max(R, axis=1)
        tmax_map[z, rr, cc] = np.argmax(R, axis=1) * TR

    if verbose:
        print("[SVD] Standard SVD complete, svd_frac =", svd_frac)
    return cbf_map, tmax_map


# ==============================================================================
# oSVD  (delay-insensitive)
# ==============================================================================

def DSC_oSVD(
    conc,
    aif_curve,
    mask,
    TR=1.5,
    aif_time=None,
    svd_frac=0.02,
    max_delay_s=16.0,       # max physiologic bolus delay to search (seconds)
                            # 16s covers clinical stroke range; reduce for speed
    verbose=True,
):
    """
    Delay-insensitive oSVD deconvolution (Ostergaard 1996 / Wu 2003).

    For each voxel, searches circular time shifts of the AIF over
    [0, max_delay_s] to find the shift that maximises the residue peak.
    This makes CBF estimation robust to bolus arrival delay.

    Speed levers
    ------------
    max_delay_s : float
        Reduce to 8.0 for quick tests (~half the shifts to search).
        Increase to 24.0 for very delayed strokes. Default 16s is safe.
    svd_frac : float
        Regularisation threshold. Higher = faster + more regularised.

    How it works
    ------------
    Standard SVD builds Toeplitz(AIF) — assumes AIF arrives at t=0.
    oSVD builds Circulant(AIF shifted by d frames) for each candidate
    delay d. The shift that gives the highest CBF (max of residue) is
    selected. The Ginv matrices are precomputed once per slice (not per
    voxel), so the per-voxel cost is just a matrix-vector multiply.
    """
    conc = np.asarray(conc, float)
    nZ, nY, nX, nT = conc.shape
    mask_rs = _resample_mask_to_conc(mask, (nZ, nY, nX))

    if isinstance(aif_curve, dict):
        aif_vec      = np.asarray(aif_curve["gv"],   float)
        aif_time_vec = np.asarray(aif_curve["time"], float)
    else:
        aif_vec      = np.asarray(aif_curve, float)
        aif_time_vec = None

    t_vox = np.arange(nT) * TR
    aif_use = (_resample_to_time(aif_vec, aif_time_vec, t_vox)
               if aif_time_vec is not None else aif_vec)

    # Number of delay frames to search
    max_delay_frames = max(1, int(np.round(max_delay_s / TR)))
    max_delay_frames = min(max_delay_frames, nT // 2)   # never search >50% of scan
    shifts = np.arange(0, max_delay_frames + 1)

    if verbose:
        print(f"[oSVD] Searching {len(shifts)} delay shifts "
              f"(0 – {max_delay_frames * TR:.1f}s), svd_frac={svd_frac}")

    # Precompute one Ginv per shift  (nShifts matrices, each nT×nT)
    # This is the key speed optimisation — done once, reused for all voxels
    Ginv_list = []
    for d in shifts:
        G = _build_circulant(aif_use, d)
        Ginv_list.append(_pseudoinverse(G, svd_frac))

    cbf_map  = np.zeros((nZ, nY, nX))
    tmax_map = np.zeros((nZ, nY, nX))

    for z in range(nZ):
        rr, cc = np.where(mask_rs[z])
        nVox = len(rr)
        if nVox == 0:
            continue

        C = conc[z, rr, cc, :].T      # (nT, nVox)

        best_cbf  = np.zeros(nVox)
        best_tmax = np.zeros(nVox)

        for d_idx, d in enumerate(shifts):
            Ginv = Ginv_list[d_idx]
            R = (1.0 / TR) * (Ginv @ C)   # (nT, nVox)
            R = R.T                        # (nVox, nT)

            # Light smoothing on residue (3-point moving average)
            if nT >= 3:
                R[:, 1:-1] = (R[:, :-2] + R[:, 1:-1] + R[:, 2:]) / 3.0
            R = np.maximum(R, 0)

            peak_vals = np.max(R, axis=1)                    # (nVox,)
            peak_idx  = np.argmax(R, axis=1).astype(float)   # (nVox,)

            # Update voxels where this shift gives higher CBF
            improved = peak_vals > best_cbf
            best_cbf[improved]  = peak_vals[improved]
            # Tmax = time of residue peak − applied delay
            # (delay is subtracted because we shifted the AIF forward)
            best_tmax[improved] = np.maximum(
                0.0,
                peak_idx[improved] * TR - d * TR
            )

        cbf_map[z, rr, cc]  = best_cbf
        tmax_map[z, rr, cc] = best_tmax

        if verbose and (z % 5 == 0 or z == nZ - 1):
            print(f"[oSVD] slice {z+1}/{nZ} done")

    if verbose:
        valid = mask_rs & (cbf_map > 0)
        print(f"[oSVD] CBF pos fraction: {valid.mean():.3f}")
        print(f"[oSVD] Tmax median: {np.median(tmax_map[mask_rs]):.2f}s  "
              f"p90: {np.percentile(tmax_map[mask_rs], 90):.2f}s")

    return cbf_map, tmax_map


# ==============================================================================
# ADAPTIVE oSVD  (oscillation-index regularization, PyPeT-style)
# ==============================================================================

# Candidate svd_frac levels tried per voxel: finest → coarsest.
# Per-voxel we pick the finest level whose residue OI ≤ OI_THRESHOLD.
_ADAPTIVE_FRACS = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
OI_THRESHOLD    = 0.10     # standard value from Wu et al. 2003


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
    Delay-insensitive oSVD with per-voxel oscillation-index adaptive
    regularization (Wu et al. 2003 / PyPeT Borghouts & Su 2025).

    Instead of a fixed svd_frac, each voxel independently chooses the
    finest (least-regularised) threshold whose residue oscillation index
    OI = Σ|dR| / (2*(nT-1)*max(R))  ≤  oi_threshold (default 0.10).

    Precomputes Ginv for every (delay_shift × svd_frac_level) pair once,
    then for each voxel selects the best shift and finest acceptable frac
    with a fully vectorised gather — no Python loops over voxels.
    """
    conc = np.asarray(conc, float)
    nZ, nY, nX, nT = conc.shape
    mask_rs = _resample_mask_to_conc(mask, (nZ, nY, nX))

    if isinstance(aif_curve, dict):
        aif_vec      = np.asarray(aif_curve["gv"],   float)
        aif_time_vec = np.asarray(aif_curve["time"], float)
    else:
        aif_vec      = np.asarray(aif_curve, float)
        aif_time_vec = None

    t_vox   = np.arange(nT) * TR
    aif_use = (_resample_to_time(aif_vec, aif_time_vec, t_vox)
               if aif_time_vec is not None else aif_vec)

    max_delay_frames = max(1, int(np.round(max_delay_s / TR)))
    max_delay_frames = min(max_delay_frames, nT // 2)
    shifts = np.arange(0, max_delay_frames + 1)
    nShifts = len(shifts)
    nFracs  = len(_ADAPTIVE_FRACS)

    if verbose:
        print(f"[oSVD-adaptive] {nShifts} shifts × {nFracs} frac levels, "
              f"OI threshold={oi_threshold}")

    # Precompute Ginv[shift, frac] — shape (nShifts, nFracs, nT, nT)
    Ginv = np.zeros((nShifts, nFracs, nT, nT), dtype=float)
    for di, d in enumerate(shifts):
        G = _build_circulant(aif_use, d)
        U, S, Vt = _svd_factors(G)
        for fi, frac in enumerate(_ADAPTIVE_FRACS):
            Ginv[di, fi] = _apply_threshold(U, S, Vt, frac)

    cbf_map  = np.zeros((nZ, nY, nX))
    tmax_map = np.zeros((nZ, nY, nX))

    for z in range(nZ):
        rr, cc = np.where(mask_rs[z])
        nVox = len(rr)
        if nVox == 0:
            continue

        C = conc[z, rr, cc, :].T    # (nT, nVox)

        # R[shift, frac, vox, t] — compute all at once
        # To avoid ~(11×7×nVox×35) float64 allocation for large slices,
        # process one shift at a time and keep only the running best.
        best_cbf  = np.zeros(nVox)
        best_tmax = np.zeros(nVox)

        for di, d in enumerate(shifts):
            # R_all: (nFracs, nVox, nT)
            R_all = np.zeros((nFracs, nVox, nT))
            for fi in range(nFracs):
                R = (1.0 / TR) * (Ginv[di, fi] @ C).T   # (nVox, nT)
                if nT >= 3:
                    R[:, 1:-1] = (R[:, :-2] + R[:, 1:-1] + R[:, 2:]) / 3.0
                R_all[fi] = np.maximum(R, 0)

            # OI for each (frac, vox)
            OI_all = np.zeros((nFracs, nVox))
            for fi in range(nFracs):
                OI_all[fi] = _oscillation_index(R_all[fi])

            # Per-voxel: finest frac index where OI ≤ threshold
            meets     = OI_all <= oi_threshold          # (nFracs, nVox)
            first_ok  = np.argmax(meets, axis=0)        # (nVox,) — 0 if none True
            no_meet   = ~meets.any(axis=0)
            first_ok[no_meet] = nFracs - 1              # coarsest if nothing works

            # Gather residues using selected frac per voxel
            R_chosen = R_all[first_ok, np.arange(nVox), :]   # (nVox, nT)

            peak_vals = R_chosen.max(axis=1)
            peak_idx  = R_chosen.argmax(axis=1).astype(float)
            improved  = peak_vals > best_cbf
            best_cbf[improved]  = peak_vals[improved]
            best_tmax[improved] = np.maximum(
                0.0, peak_idx[improved] * TR - d * TR
            )

        cbf_map[z, rr, cc]  = best_cbf
        tmax_map[z, rr, cc] = best_tmax

        if verbose and (z % 5 == 0 or z == nZ - 1):
            print(f"[oSVD-adaptive] slice {z+1}/{nZ} done")

    if verbose:
        valid = mask_rs & (cbf_map > 0)
        print(f"[oSVD-adaptive] CBF pos fraction: {valid.mean():.3f}")
        print(f"[oSVD-adaptive] Tmax median: {np.median(tmax_map[mask_rs]):.2f}s  "
              f"p90: {np.percentile(tmax_map[mask_rs], 90):.2f}s")

    return cbf_map, tmax_map
import numpy as np


def _match_mask_to_volumes(mask, volumes):
    """
    Ensure mask.shape[:3] == volumes.shape[:3] by nearest-neighbor resizing.
    Works on numpy arrays only (no physical spacing). If shapes already
    match, returns mask unchanged.
    """
    zt, rt, ct = volumes.shape[:3]
    zm, rm, cm = mask.shape[:3]
    if (zm, rm, cm) == (zt, rt, ct):
        return mask

    # Compute scale factors per axis
    sz = zt / float(zm)
    sr = rt / float(rm)
    sc = ct / float(cm)

    # Nearest-neighbor resize using scipy (safe for labels/binary masks)
    try:
        from scipy.ndimage import zoom
    except ImportError as e:
        raise ImportError(
            "scipy is required to resize the mask. pip install scipy"
        ) from e

    resized = zoom(
        mask.astype(np.float32), (sz, sr, sc), order=0, prefilter=False
    )
    # Binarize back (handles non-binary masks too)
    if resized.dtype != bool:
        resized = (resized > 0.5).astype(mask.dtype)
    else:
        resized = resized.astype(bool)
    return resized


def detect_cta_artifact(ctc, mask):
    """
    Detect and return the frame where a late CTA injection artifact begins.

    In combined CTP+CTA protocols, a second contrast bolus is injected for
    the CTA acquisition. This arrives in the brain as a massive late spike,
    completely overwhelming the first-pass CTP signal. We detect it by
    looking for a second large rise in global mean CTC after the first-pass
    peak has substantially decayed.

    Returns
    -------
    cta_start : int
        Index of first artifact frame, or ctc.shape[-1] if none detected.
    """
    n_mask = mask.sum()
    if n_mask < 10 or ctc.shape[-1] < 10:
        return ctc.shape[-1]

    # Global mean per frame (masked voxels only)
    gm = np.array(
        [ctc[mask, t].mean() for t in range(ctc.shape[-1])], dtype=float
    )

    # Use only the first 50% of frames to find the first-pass peak,
    # so a late CTA injection (which can be larger than the first pass)
    # doesn't masquerade as the peak and disable the detection.
    first_pass_end = max(len(gm) // 2, 5)
    peak_frame = int(np.argmax(gm[:first_pass_end]))

    # Post-peak reference: minimum of a 3-frame rolling mean in the settled
    # portion of the post-peak signal (up to 70% of scan length).
    # Using a fixed early window fails when the first-pass tail is broad
    # (e.g. recirculation hump), inflating the reference and raising the
    # detection threshold above the actual CTA artifact level.
    settle_start = first_pass_end  # start looking after the 50% boundary
    settle_end = max(settle_start + 3, int(len(gm) * 0.70))
    if settle_end - settle_start >= 3:
        windows = [
            gm[i : i + 3].mean() for i in range(settle_start, settle_end - 2)
        ]
        post_peak_ref = max(min(windows), 1.0)
    else:
        post_peak_ref = max(gm[settle_start : settle_start + 3].mean(), 1.0)

    # Scan for a second rise that exceeds 3x the post-peak level AND
    # is larger than 50% of the first-pass peak (clearly a second injection).
    # Start after first_pass_end to avoid triggering on the descending limb
    # of the first pass or a recirculation hump within the first half.
    search_from = first_pass_end
    for t in range(search_from, len(gm)):
        if gm[t] > 3.0 * post_peak_ref and gm[t] > 0.5 * gm[peak_frame]:
            # Walk backwards to find where the rise started
            onset = t
            while onset > search_from and gm[onset - 1] > 1.5 * post_peak_ref:
                onset -= 1
            print(
                f"[CTC] CTA artifact detected at frame {t} "
                f"(global mean {gm[t]:.1f} vs post-peak ref {post_peak_ref:.1f}). "
                f"Zeroing from onset frame {onset}."
            )
            return onset

    return ctc.shape[-1]


def DSC_conc(volumes, mask):
    # Make shapes compatible first
    mask = _match_mask_to_volumes(mask, volumes)

    S0map, bolus = DSC_ct_S0(volumes, mask)
    conc = np.zeros_like(volumes, dtype=volumes.dtype)

    ind = np.where(S0map != 0)  # compute only where mask>0
    nT = volumes.shape[3]
    for t in range(nT):
        conc[ind[0], ind[1], ind[2], t] = (
            volumes[ind[0], ind[1], ind[2], t] - S0map[ind[0], ind[1], ind[2]]
        )

    return np.real(conc), bolus


def DSC_ct_S0(volumes, mask):
    """
    Compute baseline S0 map slice-wise using a stability rule,
    with mask safely matched to the volume size.
    """
    # Make shapes compatible (no-op if already matched)
    mask = _match_mask_to_volumes(mask, volumes)

    nSamplesMin = 0
    nSamplesMax = min(volumes.shape[3] - 1, 29)  # guard against short series
    thresh = 0.01

    nS, nR, nC, nT = volumes.shape

    # global mean curve (NaN-safe)
    sig_avg = np.array(
        [np.nanmean(volumes[:, :, :, t]) for t in range(nT)], dtype=float
    )

    # per-slice mean curves
    mean_signal = np.zeros((nS, nT), dtype=float)
    for s in range(nS):
        # use only in-mask voxels for the slice mean
        m = mask[s, :, :].astype(bool)
        if not np.any(m):
            mean_signal[s, :] = np.nan  # mark empty
            continue
        for t in range(nT):
            vals = volumes[s, :, :, t]
            mean_signal[s, t] = np.nanmean(vals[m])

    # find first non-NaN slice to start
    start_s = 0
    while start_s < nS and np.isnan(np.mean(mean_signal[start_s, :])):
        start_s += 1

    S0map = np.zeros(volumes.shape[:3], dtype=volumes.dtype)
    bolus = np.zeros((nS,), dtype=int)

    for s in range(start_s, nS):
        m = mask[s, :, :].astype(bool)
        if not np.any(m):
            bolus[s] = 0
            continue

        # walk forward until the slice-mean stops changing by > thresh
        pos = nSamplesMin
        while True:
            pos = min(pos, nT - 1)
            # avoid empty window
            if pos < 1:
                pos = 1
            mean_val = np.mean(mean_signal[s, 0 : pos + 1])
            # guard divide-by-zero
            denom = mean_val if mean_val != 0 else 1e-6
            if abs((mean_val - mean_signal[s, pos]) / denom) < thresh:
                pos += 1
            else:
                pos -= 1
                break
            # stop conditions
            if pos >= nSamplesMax or pos + 1 > nT:
                pos -= 1
                break

        pos = max(1, min(pos, nT - 1))
        # Baseline S0 as the mean over frames [0:pos] (in-mask only)
        S0_slice = np.mean(volumes[s, :, :, 0:pos], axis=2)
        # apply mask to keep non-brain at zero
        S0map[s, :, :] = (S0_slice * m).astype(volumes.dtype)
        bolus[s] = pos

    return S0map, bolus

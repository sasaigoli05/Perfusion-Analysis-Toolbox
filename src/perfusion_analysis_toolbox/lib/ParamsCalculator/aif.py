import json
import numpy as np
from numpy import trapezoid
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ==============================================================================
#                             MAIN AIF FUNCTION
# ==============================================================================

# Frame interval fallback, in seconds.
#
# This used to be the pipeline's only source of timing: every study was
# assumed to sample at 1.5s. Any cohort that happens to sample at 1.5s looks
# perfectly healthy, and any other protocol silently gets wrong AIF lead-time
# gates, wrong Tmax and wrong CBF while still producing plausible maps. TR is
# now measured from the DICOM acquisition times and passed in by the caller;
# this remains only so a direct call to this module without timing is
# obviously wrong rather than subtly wrong.
FALLBACK_TR = 1.5
N_TOP_AIF = 30
MAX_REFINEMENT_ITERS = 40

# Physiological clamp ranges for the gamma-variate shape parameters. The fit
# sanity gate rejects any solution sitting on one of these bounds: a parameter
# pinned to its limit means the optimiser ran out of room rather than
# converging, and the resulting curve is not a real bolus.
ALPHA_RANGE = (1.0, 15.0)
BETA_RANGE = (0.3, 10.0)

# A fitted AIF peak this far from the measured peak means the amplitude has
# collapsed (or blown up). Deconvolving against such a curve produces a
# meaningless CBF scale, so those candidates are excluded from selection.
PEAK_RATIO_RANGE = (0.5, 2.0)

# Tolerance for deciding a fitted parameter is sitting on its bound.
BOUND_TOL = 1e-3

# Peak enhancement above this is not blood. Arterial enhancement in CTP peaks
# in the low hundreds of Hounsfield units; far above that means bone,
# calcification, or a bone edge swept through the voxel by head motion. The
# existing amplitude window is relative to the candidate pool's own median,
# so when the pool is the vessel mask it scales up with the artifact it is
# supposed to exclude. This bound does not.
MAX_ARTERIAL_PEAK_HU = 800.0

# Contrast reaches a feeding artery before the tissue it supplies, so a valid
# AIF peaks ahead of the parenchymal peak -- typically by a few seconds.
# A candidate peaking at or after the tissue is venous, parenchymal, or
# mistimed, and deconvolving against it forces every residue function to peak
# at t=0, which collapses Tmax to zero across the brain. A candidate peaking
# implausibly early is an artifact that happened to spike before the bolus.
AIF_LEAD_RANGE_S = (1.0, 8.0)

# Rather than drop the lead-time gate when few voxels pass it, accept a
# smaller candidate pool. The gate is only abandoned below this count.
MIN_AIF_CANDIDATES = 5

# ------------------------------------------------------------------------
# Region-based candidate selection
# ------------------------------------------------------------------------
# Scoring individual voxels gives an artifact no way to disqualify itself: a
# bone edge swept through one voxel by head motion produces a large, early,
# narrow curve and wins on every voxel-wise criterion there is. Requiring
# candidates to be connected components of plausible size instead means such
# a voxel has to be part of a coherent vascular structure to be considered at
# all, and the curve that gets fitted is a regional mean rather than one
# noisy time series.

# In-plane morphology applied to the seed mask before labelling: dilate to
# bridge gaps along a vessel, then erode to shed the single-voxel specks
# that survive thresholding.
REGION_DILATE_RADIUS = 3
REGION_ERODE_RADIUS = 3

# Plausible size for an arterial region, in voxels. Deliberately loose: at
# 0.39 mm in-plane an ICA cross-section alone is ~80 voxels per slice. A
# tight ceiling rejects the real arterial component and leaves only the
# fragments around it, which is worse than no ceiling at all. The curve-level
# gates below are what actually decide whether a region is arterial.
REGION_VOLUME_RANGE = (10, 3000)

# The mean curve of an arterial region must out-peak the parenchyma by this
# factor. Voxel-level gating cannot enforce this, because the gates are
# applied per voxel while the curve that gets fitted is a regional mean --
# an average of gated voxels can still be grey matter.
REGION_MIN_PEAK_RATIO = 3.0

# How many surviving regions are handed to the gamma-variate fitter.
N_TOP_REGIONS = 10

# Sum of squared second differences of the min-max normalised mean curve,
# per sample. An arterial first pass is a smooth single rise and fall; a
# curve that fails this is bimodal, spiky, or noise. Relaxed progressively
# when nothing passes at the strict value.
SMOOTHNESS_THRESHOLD = 0.04
SMOOTHNESS_RELAXATION = (1.0, 2.0, 5.0, 10.0)

# Area-under-curve percentile used to seed regions, strictest first. The TTP
# threshold is the complementary percentile, so relaxing one relaxes both.
AUC_PERCENTILE_LADDER = (95, 85, 75, 65, 55, 45, 35, 25, 15, 5)

# In-plane erosion of the brain mask before seeding, as a fraction of the
# in-plane matrix size. Skull left inside the mask by imperfect masking or
# residual motion sits at the rim; eroding it away is a spatial defence
# against the same artifact MAX_ARTERIAL_PEAK_HU catches by intensity.
BRAIN_ERODE_FRACTION = 15


def _curve_smoothness(curve):
    """
    Roughness of a curve: mean squared second difference after min-max
    normalisation, so the result does not depend on amplitude.

    Lower is smoother. A clean arterial first pass sits well below
    SMOOTHNESS_THRESHOLD; a bimodal or noise-dominated curve does not.
    Returns infinity for curves too short or too flat to judge.
    """
    curve = np.asarray(curve, dtype=float).reshape(-1)
    if curve.size < 3:
        return float("inf")
    span = float(curve.max() - curve.min())
    if span <= 1e-9:
        return float("inf")
    normalised = (curve - curve.min()) / span
    return float(np.sum(np.diff(normalised, n=2) ** 2) / curve.size)


def _eroded_brain(mask):
    """
    Brain mask eroded in-plane, to keep skull out of AIF candidacy.

    Erosion is in-plane only: slices are thick relative to the in-plane
    resolution, and eroding through them would remove whole slices at the
    vertex and skull base -- which is where the arteries are.
    """
    n_iter = max(1, int(mask.shape[2] // BRAIN_ERODE_FRACTION) // 2)
    return binary_erosion(
        mask, structure=np.ones((1, 3, 3), dtype=bool), iterations=n_iter
    )


def _region_candidates(
    conc,
    mask,
    vessel_mask=None,
    tissue_ttp=None,
    tissue_peak=None,
    TR=FALLBACK_TR,
):
    """
    Find connected arterial regions and return their mean concentration curves.

    Seeds on voxels with high area-under-curve and early time-to-peak, cleans
    the seed mask morphologically, labels connected components, and keeps the
    components that are of plausible size and whose mean curve is smooth
    enough to be a first pass. Thresholds are relaxed along a ladder until
    something survives; the physiological gates (amplitude, lead time) are
    never relaxed.

    Returns
    -------
    coords : list of (z, r, c) region centroids
    curves : (n_regions, nT) mean curve per region
    gates : dict of diagnostics, or None if no region survived
    """
    nZ, nY, nX, nT = conc.shape
    brain = _eroded_brain(np.asarray(mask, dtype=bool))
    if not brain.any():
        brain = np.asarray(mask, dtype=bool)

    peaks = conc.max(axis=-1)
    ttp = conc.argmax(axis=-1)
    auc = conc.sum(axis=-1)

    amp_ok = peaks <= MAX_ARTERIAL_PEAK_HU
    if tissue_ttp is not None:
        lead_s = (float(tissue_ttp) - ttp) * float(TR)
        lead_ok = (lead_s >= AIF_LEAD_RANGE_S[0]) & (
            lead_s <= AIF_LEAD_RANGE_S[1]
        )
    else:
        lead_ok = np.ones(peaks.shape, dtype=bool)

    searchable = brain & amp_ok & lead_ok
    gates = {
        "mode": "region",
        "n_candidates": int(brain.sum()),
        "rejected_amplitude": int((brain & ~amp_ok).sum()),
        "rejected_lead_time": int((brain & ~lead_ok).sum()),
        "tissue_ttp": None if tissue_ttp is None else int(tissue_ttp),
        "lead_gate_dropped": False,
    }
    print(
        f"[AIF] Region search: {int(searchable.sum())}/{int(brain.sum())} "
        f"eroded-brain voxels pass the physiological gates "
        f"(rejected {gates['rejected_amplitude']} on peak > "
        f"{MAX_ARTERIAL_PEAK_HU:.0f} HU, {gates['rejected_lead_time']} on "
        f"lead time)"
    )
    if not searchable.any():
        return [], np.zeros((0, nT)), None

    auc_pool = auc[searchable]
    ttp_pool = ttp[searchable]
    dil = np.ones((1, REGION_DILATE_RADIUS, REGION_DILATE_RADIUS), dtype=bool)
    ero = np.ones((1, REGION_ERODE_RADIUS, REGION_ERODE_RADIUS), dtype=bool)
    vol_lo, vol_hi = REGION_VOLUME_RANGE

    for pct in AUC_PERCENTILE_LADDER:
        auc_thr = float(np.percentile(auc_pool, pct))
        ttp_thr = float(np.percentile(ttp_pool, 100 - pct))
        seed = searchable & (auc > auc_thr) & (ttp <= ttp_thr) & (ttp > 0)
        if not seed.any():
            continue

        seed = binary_erosion(binary_dilation(seed, dil), ero)
        labels, n_regions = label(seed)
        if n_regions == 0:
            continue

        for relax in SMOOTHNESS_RELAXATION:
            limit = SMOOTHNESS_THRESHOLD * relax
            found = []
            for idx in range(1, n_regions + 1):
                region = labels == idx
                vol = int(region.sum())
                if vol < vol_lo or vol > vol_hi:
                    continue
                curve = conc[region].mean(axis=0).astype(float)

                # Gate the curve that will actually be fitted, not only the
                # voxels it came from. A mean over voxels that each pass the
                # lead-time test can still peak somewhere else entirely, and
                # an average of gated voxels can still be parenchyma.
                if tissue_peak is not None and curve.max() < (
                    REGION_MIN_PEAK_RATIO * tissue_peak
                ):
                    continue
                if tissue_ttp is not None:
                    region_lead = (
                        float(tissue_ttp) - int(np.argmax(curve))
                    ) * float(TR)
                    if not (
                        AIF_LEAD_RANGE_S[0]
                        <= region_lead
                        <= AIF_LEAD_RANGE_S[1]
                    ):
                        continue

                smoothness = _curve_smoothness(curve)
                if smoothness > limit:
                    continue

                # Favours curves that come back down after the peak. A
                # venous or parenchymal curve plateaus instead, and a
                # first-pass arterial curve is what deconvolution needs.
                washout = float(curve.max() - curve[-3:].mean())
                if washout <= 0:
                    continue

                zz, rr, cc = np.nonzero(region)
                found.append(
                    {
                        "coord": (
                            int(round(zz.mean())),
                            int(round(rr.mean())),
                            int(round(cc.mean())),
                        ),
                        "curve": curve,
                        "vol": vol,
                        "smoothness": smoothness,
                        "score": vol * washout / (smoothness + 1e-9),
                    }
                )

            if found:
                found.sort(key=lambda r: r["score"], reverse=True)
                found = found[:N_TOP_REGIONS]
                gates.update(
                    auc_percentile=pct,
                    smoothness_limit=float(limit),
                    smoothness_relaxed=bool(relax > 1.0),
                    n_regions_labelled=int(n_regions),
                    n_selected=len(found),
                    region_volumes=[r["vol"] for r in found],
                    region_smoothness=[
                        round(r["smoothness"], 5) for r in found
                    ],
                )
                print(
                    f"[AIF] {len(found)} region(s) accepted at AUC "
                    f"percentile {pct}, smoothness <= {limit:.3f} "
                    f"(of {n_regions} labelled); volumes "
                    f"{[r['vol'] for r in found]}"
                )
                return (
                    [r["coord"] for r in found],
                    np.stack([r["curve"] for r in found]),
                    gates,
                )

    print(
        "[AIF] No connected region satisfied the volume and smoothness "
        "constraints at any threshold — falling back to voxel scoring."
    )
    return [], np.zeros((0, nT)), None


def _global_arterial_score(
    conc,
    mask,
    n_top=N_TOP_AIF,
    vessel_mask=None,
    tissue_ttp=None,
    TR=FALLBACK_TR,
):
    """
    Score all brain voxels globally and return top-N most arterial candidates.

    If vessel_mask is provided and contains at least n_top voxels within the
    brain mask, restricts the candidate pool to those voxels (high confidence
    arterial voxels from preprocessing). Falls back to the full brain if too few.

    tissue_ttp, when given, is the frame at which the parenchyma peaks.
    Candidates are then required to lead it by AIF_LEAD_RANGE_S.

    Returns
    -------
    coords : list of (z, r, c) tuples — one per selected voxel
    curves : np.ndarray of shape (n_top, nT) — corresponding CTC curves
    gates : dict describing which physiological gates rejected what
    """
    nZ, nY, nX, nT = conc.shape

    if vessel_mask is not None:
        vessel_in_mask = vessel_mask & mask
        n_vessel = int(vessel_in_mask.sum())
        print(f"[AIF] Vessel mask: {n_vessel} voxels inside brain mask")
        if n_vessel >= n_top:
            print(
                f"[AIF] Restricting AIF search to vessel mask ({n_vessel} voxels)"
            )
            zz, rr, cc = np.where(vessel_in_mask)
        else:
            print(
                f"[AIF] Too few vessel-mask voxels ({n_vessel}) — using full brain"
            )
            zz, rr, cc = np.where(mask)
    else:
        zz, rr, cc = np.where(mask)
    if len(zz) == 0:
        return [], np.zeros((0, nT)), {"n_candidates": 0}

    all_curves = conc[zz, rr, cc, :]  # (N, nT)
    peaks = all_curves.max(axis=1)
    ttp = np.argmax(all_curves, axis=1).astype(float)

    # Two-sided amplitude filter targeting pseudo-arterial voxels:
    #   lower: require > 2× median to skip pure parenchyma
    #   upper: cap at 6× median to exclude bone/artifact whose S0 was mis-estimated
    # Together this targets voxels with elevated first-pass enhancement that are
    # plausibly vascular without being bone or dural sinus artifacts.
    # If the range is too narrow (< n_top candidates), fall back to all voxels ≤ 6× median.
    pos_peaks = peaks[peaks > 0]
    peak_median = float(np.median(pos_peaks)) if len(pos_peaks) > 0 else 1.0
    peak_lo = 2.0 * peak_median
    peak_hi = 6.0 * peak_median
    print(
        f"[AIF] CTC peak median={peak_median:.1f}  arterial window=[{peak_lo:.1f}, {peak_hi:.1f}]"
    )

    ttp_min = max(2, int(0.05 * nT))
    ttp_max = int(0.75 * nT)

    # Absolute physiological gates. The relaxation ladder below never drops
    # these: they are what separates an artery from a bone edge or a vein,
    # and relaxing them is how an artifact wins selection.
    amp_ok = peaks <= MAX_ARTERIAL_PEAK_HU

    if tissue_ttp is not None:
        lead_s = (float(tissue_ttp) - ttp) * float(TR)
        lead_ok = (lead_s >= AIF_LEAD_RANGE_S[0]) & (
            lead_s <= AIF_LEAD_RANGE_S[1]
        )
    else:
        lead_ok = np.ones(peaks.shape, dtype=bool)

    hard = amp_ok & lead_ok
    gates = {
        "mode": "voxel",
        "n_candidates": int(len(zz)),
        "rejected_amplitude": int((~amp_ok).sum()),
        "rejected_lead_time": int((~lead_ok).sum()),
        "tissue_ttp": None if tissue_ttp is None else int(tissue_ttp),
        "lead_gate_dropped": False,
    }
    print(
        f"[AIF] Physiological gates: {int(hard.sum())}/{len(zz)} pass "
        f"(rejected {gates['rejected_amplitude']} on peak > "
        f"{MAX_ARTERIAL_PEAK_HU:.0f} HU, {gates['rejected_lead_time']} on "
        f"lead time outside {AIF_LEAD_RANGE_S[0]:.0f}-"
        f"{AIF_LEAD_RANGE_S[1]:.0f}s)"
    )

    if hard.sum() < MIN_AIF_CANDIDATES and tissue_ttp is not None:
        print(
            f"[AIF] WARNING: only {int(hard.sum())} voxel(s) lead the "
            f"parenchymal peak by {AIF_LEAD_RANGE_S[0]:.0f}-"
            f"{AIF_LEAD_RANGE_S[1]:.0f}s. Dropping the lead-time gate — the "
            "resulting AIF is mistimed and Tmax from it is not meaningful."
        )
        hard = amp_ok
        gates["lead_gate_dropped"] = True

    valid = hard & (ttp >= ttp_min) & (ttp <= ttp_max) & (peaks >= peak_lo)
    valid &= peaks <= peak_hi
    print(
        f"[AIF] Global search: {int(valid.sum())} / {len(zz)} brain voxels pass TTP+arterial window"
    )

    if valid.sum() < MIN_AIF_CANDIDATES:
        print(
            f"[AIF] Too few candidates — relaxing to all peaks ≤ {peak_hi:.1f}"
        )
        valid = (hard & (ttp >= ttp_min) & (ttp <= ttp_max) & (peaks > 0)) & (
            peaks <= peak_hi
        )

    if valid.sum() < MIN_AIF_CANDIDATES:
        print("[AIF] Warning: few voxels pass TTP window — relaxing it too")
        valid = hard & (peaks > 0)

    gates["n_passing"] = int(valid.sum())

    v_zz = zz[valid]
    v_rr = rr[valid]
    v_cc = cc[valid]
    v_curves = all_curves[valid]
    v_peaks = peaks[valid]
    v_ttp = ttp[valid]

    widths = np.sum(v_curves > (v_peaks / 2.0)[:, None], axis=1).astype(float)
    widths = np.maximum(widths, 1.0)

    def _norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-9 else np.zeros_like(x)

    score = _norm01(v_peaks) - 0.4 * _norm01(v_ttp) - 0.4 * _norm01(widths)

    top_k = min(n_top, len(score))
    top_idx = np.argsort(score)[-top_k:]

    coords = list(
        zip(
            v_zz[top_idx].tolist(),
            v_rr[top_idx].tolist(),
            v_cc[top_idx].tolist(),
        )
    )
    curves = v_curves[top_idx]

    slices_used = sorted({int(v_zz[i]) for i in top_idx})
    print(f"[AIF] Global top-{top_k} candidates from slices: {slices_used}")
    gates["n_selected"] = int(top_k)
    return coords, curves, gates


def _fit_aif_global(coords, curves, savepath, TR, gates=None):
    """
    Run GV fitting on globally-selected AIF candidate voxels.
    Returns the AIF dict for the voxel with the earliest fitted t0.
    """
    nT = curves.shape[1]
    time = np.arange(nT) * TR
    n = len(coords)

    fit_params = np.zeros((n, 4), dtype=float)

    print(f"[AIF] Global GV fitting: {n} candidates...")
    for i in range(n):
        curve = curves[i].astype(float)
        weights = 0.01 + np.exp(-curve)
        TTP = int(np.argmax(curve))
        TTP = max(0, min(TTP, nT - 1))
        weights[TTP] /= 10.0
        if TTP - 1 >= 0:
            weights[TTP - 1] /= 5.0
        if TTP + 1 < nT:
            weights[TTP + 1] /= 2.0
        fit_params[i] = fitGV_peak1(curve.copy(), weights.copy(), TR, time)

    valid_ab = (fit_params[:, 1] > 0) & (fit_params[:, 2] > 0)
    if valid_ab.sum() > 0:
        avg_alpha = float(np.mean(fit_params[valid_ab, 1]))
        avg_beta = float(np.mean(fit_params[valid_ab, 2]))
    else:
        avg_alpha, avg_beta = 5.0, 1.5
    print(f"[AIF] Initial avg_alpha={avg_alpha:.3f}, avg_beta={avg_beta:.3f}")

    fit_bat = np.zeros((n, 2), dtype=float)
    fit_ab = np.zeros((n, 2), dtype=float)
    fit_temp = fit_params.copy()

    print(
        f"[AIF] Starting refinement loop (max {MAX_REFINEMENT_ITERS} iterations)..."
    )
    for nTrue in range(1, MAX_REFINEMENT_ITERS + 1):
        for i in range(n):
            curve = curves[i].astype(float)
            weights = 0.01 + np.exp(-curve)
            TTP = int(np.argmax(curve))
            TTP = max(0, min(TTP, nT - 1))
            weights[TTP] /= 10.0
            if TTP - 1 >= 0:
                weights[TTP - 1] /= 5.0
            if TTP + 1 < nT:
                weights[TTP + 1] /= 2.0
            try:
                fit_bat[i] = find_params_bat_optim(
                    curve,
                    weights,
                    [avg_alpha, avg_beta],
                    fit_temp[i, 0],
                    fit_temp[i, 3],
                    TR,
                    time,
                )
                fit_ab[i] = find_params_alpha_beta_optim(
                    curve,
                    weights,
                    fit_bat[i],
                    avg_alpha,
                    avg_beta,
                    TR,
                    time,
                )
            except Exception:
                pass
            fit_temp[i, 0] = fit_bat[i, 0]
            fit_temp[i, 3] = fit_bat[i, 1]
            fit_temp[i, 1] = fit_ab[i, 0]
            fit_temp[i, 2] = fit_ab[i, 1]

        # Clamp to physiological ranges to prevent alternating-fit runaway
        fit_temp[:, 1] = np.clip(fit_temp[:, 1], *ALPHA_RANGE)
        fit_temp[:, 2] = np.clip(fit_temp[:, 2], *BETA_RANGE)

        valid_ab = (fit_temp[:, 1] > 0) & (fit_temp[:, 2] > 0)
        if valid_ab.sum() > 0:
            avg_alpha = float(np.mean(fit_temp[valid_ab, 1]))
            avg_beta = float(np.mean(fit_temp[valid_ab, 2]))

        if nTrue % 10 == 0:
            print(
                f"[AIF] refinement iter {nTrue}: avg_alpha={avg_alpha:.3f}, avg_beta={avg_beta:.3f}"
            )

    print(
        f"[AIF] Reached MAX_REFINEMENT_ITERS={MAX_REFINEMENT_ITERS}, stopping."
    )

    # Compute per-voxel fit residual (MSE between measured curve and GV fit)
    residuals = np.zeros(n)
    for i in range(n):
        gv = GVfunction_peak1(fit_temp[i], TR, time)
        residuals[i] = float(np.mean((curves[i].astype(float) - gv) ** 2))

    # Keep voxels whose residual is below 2× the median — rejects badly-fit curves
    med_resid = float(np.median(residuals))
    resid_ok = residuals < 2.0 * med_resid

    # Fit sanity gate. The residual gate alone is relative to the candidate
    # set, so when most candidates fit poorly a degenerate solution can still
    # pass it. These checks are absolute.
    peak_meas = np.asarray(curves, dtype=float).max(axis=1)
    peak_fit = np.array(
        [
            float(GVfunction_peak1(fit_temp[i], TR, time).max())
            for i in range(n)
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_ratio = np.where(peak_meas > 0, peak_fit / peak_meas, 0.0)

    amp_ok = (peak_ratio >= PEAK_RATIO_RANGE[0]) & (
        peak_ratio <= PEAK_RATIO_RANGE[1]
    )
    alpha_ok = (fit_temp[:, 1] > ALPHA_RANGE[0] + BOUND_TOL) & (
        fit_temp[:, 1] < ALPHA_RANGE[1] - BOUND_TOL
    )
    beta_ok = (fit_temp[:, 2] > BETA_RANGE[0] + BOUND_TOL) & (
        fit_temp[:, 2] < BETA_RANGE[1] - BOUND_TOL
    )
    # t0 pinned at zero means the fit never located a bolus onset.
    t0_ok = fit_temp[:, 0] > BOUND_TOL

    sane = amp_ok & alpha_ok & beta_ok & t0_ok
    good_fit = resid_ok & sane
    fit_gate_failed = False

    if good_fit.sum() == 0:
        # Nothing passed. Fall back to residual-only so the pipeline still
        # produces a result, but say so loudly: the AIF is not trustworthy.
        print(
            "[AIF] WARNING: no candidate passed the fit sanity gate "
            f"(amp={int(amp_ok.sum())}, alpha={int(alpha_ok.sum())}, "
            f"beta={int(beta_ok.sum())}, t0={int(t0_ok.sum())} of {n}). "
            "Falling back to residual gate only — treat this AIF as suspect."
        )
        good_fit = resid_ok if resid_ok.sum() else np.ones(n, dtype=bool)
        fit_gate_failed = True

    print(
        f"[AIF] Fit quality: {int(good_fit.sum())}/{n} voxels pass "
        f"(residual={int(resid_ok.sum())}, sanity={int(sane.sum())}, "
        f"median MSE={med_resid:.4f})"
    )
    if int(sane.sum()) < n:
        print(
            f"[AIF] Rejected as degenerate: {int((~amp_ok).sum())} on amplitude, "
            f"{int((~alpha_ok).sum())} on alpha bound, "
            f"{int((~beta_ok).sum())} on beta bound, "
            f"{int((~t0_ok).sum())} on t0"
        )

    # Among good-fit voxels, pick the one with the earliest t0
    t0_vals = fit_temp[:, 0].copy()
    t0_vals[~good_fit] = np.inf
    best_i = int(np.argmin(t0_vals))
    final_params = fit_temp[best_i].copy()
    final_conc = curves[best_i].astype(float)
    best_coord = coords[best_i]

    gv_curve = GVfunction_peak1(final_params, TR, time)
    print(
        f"[AIF] Selected voxel {best_coord}: measured peak={float(final_conc.max()):.1f} HU  "
        f"GV peak={float(gv_curve.max()):.1f} HU  t0={float(final_params[0]):.2f}s"
    )

    # Diagnostic: AIF curve + fit
    fig, ax = plt.subplots(figsize=(8, 4))
    for curve in curves:
        ax.plot(time, curve, color="steelblue", alpha=0.2, linewidth=0.8)
    ax.plot(
        time,
        final_conc,
        "bo",
        markersize=5,
        label=f"Selected (z={best_coord[0]})",
    )
    ax.plot(time, gv_curve, "k-", linewidth=2.5, label="GV fit")
    t0, alpha, beta, A = final_params
    ax.set_title(
        f"AIF  t0={t0:.2f}s  α={alpha:.2f}  β={beta:.2f}  A={A:.3f}\n"
        f"peak={np.max(gv_curve):.3f}  integral={trapezoid(gv_curve, time):.3f}"
    )
    ax.set_xlabel("Time [s]")
    ax.legend()
    plt.tight_layout()
    # Diagnostics are optional: an in-memory caller passes no save path.
    if savepath:
        plt.savefig(f"{savepath}/aif_curve.png", dpi=150)
    plt.close()

    params_dict = {
        "t0_s": float(final_params[0]),
        "alpha": float(final_params[1]),
        "beta": float(final_params[2]),
        "A": float(final_params[3]),
        "peak_gv": float(np.max(gv_curve)),
        "integral_gv": float(trapezoid(gv_curve, time)),
        "TR_s": float(TR),
        "nT": int(nT),
        "best_coord": [int(x) for x in best_coord],
    }

    # Quality record. The pipeline still returns an AIF when these checks
    # fail -- refusing outright would strand a whole batch -- but the run is
    # marked suspect so downstream results are not read as if they were sound.
    if gates:
        params_dict["gates"] = gates

        # One top-level field naming the method that actually produced this
        # AIF, so a cohort can be filtered without walking into gates. A run
        # that asked for anatomical selection and fell back reads as
        # "global_arterial_score (fallback from external_mask)", never as a
        # clean anatomical result.
        fallback = gates.get("fallback")
        selection = gates.get("mode", "unknown")
        if fallback:
            selection = f"{fallback['to']} (fallback from {fallback['from']})"
        params_dict["aif_selection"] = selection
        params_dict["used_fallback"] = bool(fallback)

        # A run is suspect if any of the things that make an AIF trustworthy
        # had to be given up: its timing, the gate that enforces that timing,
        # the shape of the fit, or the smoothness requirement that separates a
        # bolus from noise. Barnes-type studies clear the timing check and fail
        # every other one, so keying the verdict on lead time alone lets them
        # through.
        #
        # A fallback is deliberately NOT a suspect reason. Voxel scoring is a
        # method in its own right and the curve it returns still passes every
        # gate below; what changed is which method ran, which `used_fallback`
        # records separately. Folding it in here would mark a usable run as a
        # failed fit.
        reasons = []
        if fit_gate_failed:
            reasons.append("no candidate passed the fit sanity gate")
        if gates.get("lead_gate_dropped"):
            reasons.append("lead-time gate dropped")
        if gates.get("smoothness_relaxed"):
            reasons.append("smoothness threshold relaxed")

        tissue_ttp = gates.get("tissue_ttp")
        if tissue_ttp is not None:
            lead = (float(tissue_ttp) - int(np.argmax(final_conc))) * TR
            in_range = AIF_LEAD_RANGE_S[0] <= lead <= AIF_LEAD_RANGE_S[1]
            params_dict["lead_time_s"] = float(lead)
            if not in_range:
                reasons.append("lead time out of range")
            verdict = "OK" if in_range else "OUT OF RANGE"
            print(
                f"[AIF] QC: selected voxel leads the parenchymal peak by "
                f"{lead:.1f}s -- {verdict} "
                f"(expected {AIF_LEAD_RANGE_S[0]:.0f}-"
                f"{AIF_LEAD_RANGE_S[1]:.0f}s)"
            )

        # Set unconditionally. Previously these keys only appeared when a
        # parenchymal reference was available, so a study with no tissue_ttp
        # produced a record with no verdict at all and read as clean.
        params_dict["suspect"] = bool(reasons)
        params_dict["suspect_reasons"] = reasons
        if params_dict["suspect"]:
            print(
                "[AIF] WARNING: AIF is SUSPECT -- "
                + "; ".join(reasons)
                + ". CBF scale and Tmax from this study should not be "
                "trusted."
            )

    if savepath:
        with open(f"{savepath}/aif_params.json", "w") as f:
            json.dump(params_dict, f, indent=2)

    print("[AIF] Final parameters:", params_dict)

    return {
        "parameters": final_params,
        "gv": gv_curve,
        "time": time,
        "conc": final_conc,
    }


def _parenchymal_reference(conc, mask, vessels):
    """
    Peak frame and peak amplitude of the parenchyma.

    These are the two things a candidate AIF is measured against: it must
    arrive earlier than the tissue it supplies, and it must be substantially
    brighter. Vessels are excluded so the reference reflects tissue rather
    than the arteries the AIF is drawn from. Returns (None, None) when there
    is no parenchyma to average, in which case both gates are skipped.
    """
    parenchyma = mask & ~vessels if vessels is not None else mask
    if not parenchyma.any():
        parenchyma = mask
    if not parenchyma.any():
        return None, None
    curve = np.array(
        [float(conc[..., t][parenchyma].mean()) for t in range(conc.shape[-1])]
    )
    return int(np.argmax(curve)), float(curve.max())


def DSC_aif(mask, conc, vessels, savepath=None, mode="voxel", TR=None,
            aif_mask=None):
    """
    Fit the arterial input function from the concentration series.

    Scores every brain voxel for arterial character (early, narrow, high peak),
    restricted to the vessel mask when it is large enough, then fits a
    gamma-variate to the best candidates. Candidates must also clear two
    physiological gates: a peak amplitude blood can actually reach, and a
    lead time ahead of the parenchymal peak.

    Returns a dict with keys 'gv' (fitted curve), 'time' and 'TR'.
    """

    # TR must come from the caller, which measures it. The fallback exists so
    # a direct call without timing is obviously wrong rather than subtly so.
    TR = float(TR) if TR else FALLBACK_TR
    tissue_ttp, tissue_peak = _parenchymal_reference(conc, mask, vessels)
    if tissue_ttp is not None:
        print(
            f"[AIF] Parenchymal peak {tissue_peak:.1f} HU at frame "
            f"{tissue_ttp} ({tissue_ttp * TR:.1f}s); AIF must lead it "
            f"by {AIF_LEAD_RANGE_S[0]:.0f}-{AIF_LEAD_RANGE_S[1]:.0f}s and "
            f"exceed {REGION_MIN_PEAK_RATIO:.0f}x its amplitude"
        )

    # Region mode requires candidates to be connected structures of
    # plausible size whose mean curve is smooth, which is the more principled
    # construction and produces far more informative failure flags. It is not
    # the default: measured against the commercial reference on the current
    # cohort it scores slightly lower than voxel scoring, and its threshold
    # ladders are not yet tuned for this resolution. Voxel scoring also acts
    # as the fallback whenever no region survives.
    coords, curves, gates = (None, None, None)
    if str(mode).lower() == "region":
        coords, curves, gates = _region_candidates(
            conc,
            mask,
            vessel_mask=vessels,
            tissue_ttp=tissue_ttp,
            tissue_peak=tissue_peak,
            TR=TR,
        )

    # An externally supplied arterial mask replaces candidate *selection*
    # only. Everything after this point -- curve extraction, the
    # gamma-variate fit, TR handling -- is unchanged, so the comparison
    # isolates where the AIF is taken from.
    fallback = None
    if aif_mask is not None:
        supplied = np.asarray(aif_mask).astype(bool)
        sel = supplied & np.asarray(mask).astype(bool)
        coords = np.argwhere(sel)
        if len(coords) == 0:
            # Vessel segmentation does not always land inside the brain mask:
            # on this cohort it returned nothing arterial in four studies and
            # lost every voxel to the intersection in a fifth. Refusing here
            # strands those studies entirely, so fall back to voxel scoring --
            # the method used before anatomical selection existed -- and record
            # that the result did not come from the mask that was asked for.
            fallback = {
                "from": "external_mask",
                "to": "global_arterial_score",
                "reason": ("supplied aif_mask is empty"
                           if not supplied.any() else
                           "supplied aif_mask selects no brain voxels"),
                "n_supplied": int(supplied.sum()),
            }
            print(f"[AIF] WARNING: {fallback['reason']} "
                  f"({fallback['n_supplied']} voxels supplied, 0 inside the "
                  f"brain mask). Falling back to voxel scoring; this study is "
                  f"not an anatomical-AIF result.")
        else:
            # conc is (Z, Y, X, T), so boolean indexing on the spatial axes
            # yields (n_voxels, nT) directly -- the same layout the scorer
            # produces and the fitter expects.
            curves = conc[sel]
            n_all = len(coords)
            if n_all > N_TOP_AIF:
                # Fitting a gamma-variate per voxel is the expensive step, and
                # a labelled vessel can hold hundreds. Keep the strongest,
                # which are the least partial-volumed within an already-correct
                # region.
                keep = np.argsort(curves.max(axis=1))[::-1][:N_TOP_AIF]
                coords, curves = coords[keep], curves[keep]
            gates = {"mode": "external_mask", "n_candidates": int(len(coords)),
                     "n_in_mask": int(n_all)}
            print(f"[AIF] external mask: {n_all} arterial voxels, "
                  f"fitting the strongest {len(coords)}")

    if gates is None:
        coords, curves, gates = _global_arterial_score(
            conc,
            mask,
            n_top=N_TOP_AIF,
            vessel_mask=vessels,
            tissue_ttp=tissue_ttp,
            TR=TR,
        )
        if fallback is not None:
            # Stamped into gates so it reaches aif_params.json, and read again
            # in _fit_aif_global to mark the run suspect.
            gates["fallback"] = fallback

    if len(coords) == 0:
        raise ValueError("[AIF] No valid AIF candidates found in brain mask.")

    print(f"[AIF] Candidate mode: {gates.get('mode', 'unknown')}")

    aif = _fit_aif_global(coords, curves, savepath, TR=TR, gates=gates)
    aif["TR"] = TR

    print(f"[AIF] Peak GV amplitude: {np.max(aif['gv']):.4f}")
    print(f"[AIF] AIF integral (trapz): {trapezoid(aif['gv']):.4f}")

    return aif


# ==============================================================================
#                   FITTING HELPER FUNCTIONS
# ==============================================================================


def sanitize_bounds(p0):
    p0 = np.asarray(p0, float)
    p0 = np.maximum(p0, 1e-6)
    lb = 0.1 * p0
    ub = 10.0 * p0
    too_close = ub - lb <= 1e-8
    ub[too_close] = lb[too_close] + 1e-6
    return lb, ub


def make_feasible_initial_guess(p0):
    p0 = np.asarray(p0, float)
    lb, ub = sanitize_bounds(p0)
    eps = 1e-6
    p0 = np.maximum(p0, lb + eps)
    p0 = np.minimum(p0, ub - eps)
    return p0, lb, ub


def fitGV_peak1(data, weights, TR, time):
    data = np.asarray(data, float)
    weights = np.asarray(weights, float)
    nT = data.shape[0]

    MCdata = np.max(data)
    TTPpos = int(np.argmax(data))
    TTPpos = max(0, min(TTPpos, nT - 1))

    if TTPpos > 0:
        idx = np.where(data[:TTPpos] <= 0.05 * MCdata)[0]
        last_idx = idx[-1] if idx.size > 0 else 0
    else:
        last_idx = 0

    t0_init = time[last_idx]
    alpha_init = 5.0
    TTPdata = time[TTPpos]
    beta_init = max((TTPdata - t0_init) / max(alpha_init, 1e-6), 1e-3)

    gv_for_A = GVfunction_peak1(
        [t0_init, alpha_init, beta_init, 1.0], TR, time
    )
    max_gv = np.max(gv_for_A)
    A_init = MCdata if max_gv <= 0 else MCdata / max_gv

    p0 = np.array([t0_init, alpha_init, beta_init, A_init], dtype=float)
    p0, lb, ub = make_feasible_initial_guess(p0)

    if 0 <= TTPpos < nT:
        weights[TTPpos] = max(weights[TTPpos] / 10.0, 1e-6)
    if TTPpos - 1 >= 0:
        weights[TTPpos - 1] = max(weights[TTPpos - 1] / 2.0, 1e-6)

    def obj(p):
        vec = GVfunction_peak1(p, TR, time)
        return (vec - data) / (weights + 1e-6)

    res = least_squares(
        obj,
        p0,
        bounds=(lb, ub),
        max_nfev=1000,
        xtol=1e-8,
        gtol=1e-8,
        ftol=1e-8,
    )
    return np.transpose(res.x)


def GVfunction_peak1(p, TR, time):
    t0, alpha, beta, A = p
    time = np.asarray(time, float)
    GV = np.zeros_like(time)
    mask = time > t0
    x = time[mask] - t0
    GV[mask] = A * (x**alpha) * np.exp(-x / max(beta, 1e-6))
    return GV


def fitGV_peak1_2(data, weights, param1, param2, t0, A, TR, time):
    data = np.asarray(data, float)
    weights = np.asarray(weights, float)
    nT = data.shape[0]

    MCdata = np.max(data)
    TTPpos = int(np.argmax(data))
    TTPpos = max(0, min(TTPpos, nT - 1))

    if TTPpos > 0:
        idx = np.where(data[:TTPpos] <= 0.05 * MCdata)[0]
        last_idx = idx[-1] if idx.size > 0 else 0
    else:
        last_idx = 0

    t0_init = time[last_idx] if t0 == 0 else t0

    if A == 0:
        gv_for_A = GVfunction_peak1_2([t0_init, 1.0], param1, param2, TR, time)
        max_gv = np.max(gv_for_A)
        A_init = MCdata if max_gv <= 0 else MCdata / max_gv
    else:
        A_init = A

    p0 = np.array([t0_init, A_init], dtype=float)
    p0, lb, ub = make_feasible_initial_guess(p0)

    def obj(p):
        vec = GVfunction_peak1_2(p, param1, param2, TR, time)
        return (vec - data) / (weights + 1e-6)

    res = least_squares(
        obj,
        p0,
        bounds=(lb, ub),
        max_nfev=1000,
        xtol=1e-8,
        gtol=1e-8,
        ftol=1e-8,
    )
    return np.transpose(res.x)


def GVfunction_peak1_2(p, param1, param2, TR, time):
    t0, A = p
    alpha = param1
    beta = param2
    time = np.asarray(time, float)
    GV = np.zeros_like(time)
    mask = time > t0
    x = time[mask] - t0
    GV[mask] = A * (x**alpha) * np.exp(-x / max(beta, 1e-6))
    return GV


def fitGV_peak1_1(data, weights, param1, param2, alpha, beta, TR, time):
    data = np.asarray(data, float)
    weights = np.asarray(weights, float)

    p0 = np.array([max(alpha, 1e-6), max(beta, 1e-6)], dtype=float)
    p0, lb, ub = make_feasible_initial_guess(p0)

    def obj(p):
        vec = GVfunction_peak1_1(p, param1, param2, TR, time)
        return (vec - data) / (weights + 1e-6)

    res = least_squares(
        obj,
        p0,
        bounds=(lb, ub),
        max_nfev=1000,
        xtol=1e-8,
        gtol=1e-8,
        ftol=1e-8,
    )
    return np.transpose(res.x)


def GVfunction_peak1_1(p, param1, param2, TR, time):
    alpha, beta = p
    t0 = param1
    A = param2
    time = np.asarray(time, float)
    GV = np.zeros_like(time)
    mask = time > t0
    x = time[mask] - t0
    GV[mask] = A * (x**alpha) * np.exp(-x / max(beta, 1e-6))
    return GV


def find_params_bat_optim(
    avg_test, avg_weights, fitParams_test, t0, A, TR, time
):
    param1 = fitParams_test[0]
    param2 = fitParams_test[1]
    return fitGV_peak1_2(
        avg_test, avg_weights, param1, param2, t0, A, TR, time
    )[0:2]


def find_params_alpha_beta_optim(
    avg_test, avg_weights, fitParams_test, alpha, beta, TR, time
):
    param1 = fitParams_test[0]
    param2 = fitParams_test[1]
    return fitGV_peak1_1(
        avg_test,
        avg_weights,
        param1,
        param2,
        alpha,
        beta,
        TR,
        time,
    )[0:2]

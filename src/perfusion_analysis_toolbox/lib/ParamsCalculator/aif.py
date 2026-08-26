import json
import numpy as np
from numpy import trapezoid
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ==============================================================================
#                             MAIN AIF FUNCTION
# ==============================================================================

FIXED_TR = 1.5
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


def _global_arterial_score(conc, mask, n_top=N_TOP_AIF, vessel_mask=None):
    """
    Score all brain voxels globally and return top-N most arterial candidates.

    If vessel_mask is provided and contains at least n_top voxels within the
    brain mask, restricts the candidate pool to those voxels (high confidence
    arterial voxels from preprocessing). Falls back to the full brain if too few.

    Returns
    -------
    coords : list of (z, r, c) tuples — one per selected voxel
    curves : np.ndarray of shape (n_top, nT) — corresponding CTC curves
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
        return [], np.zeros((0, nT))

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
    valid = (
        (ttp >= ttp_min)
        & (ttp <= ttp_max)
        & (peaks >= peak_lo)
        & (peaks <= peak_hi)
    )
    print(
        f"[AIF] Global search: {int(valid.sum())} / {len(zz)} brain voxels pass TTP+arterial window"
    )

    if valid.sum() < n_top:
        # Relax lower bound — keep upper bound to still exclude bone
        print(
            f"[AIF] Too few candidates — relaxing to all peaks ≤ {peak_hi:.1f}"
        )
        valid = (
            (ttp >= ttp_min)
            & (ttp <= ttp_max)
            & (peaks > 0)
            & (peaks <= peak_hi)
        )

    if valid.sum() < n_top:
        print(
            "[AIF] Warning: few voxels pass TTP window — using all brain voxels"
        )
        valid = peaks > 0

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
    return coords, curves


def _fit_aif_global(coords, curves, savepath, TR):
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
    with open(f"{savepath}/aif_params.json", "w") as f:
        json.dump(params_dict, f, indent=2)

    print("[AIF] Final parameters:", params_dict)

    return {
        "parameters": final_params,
        "gv": gv_curve,
        "time": time,
        "conc": final_conc,
    }


def DSC_aif(mask, conc, vessels, savepath):
    """
    Fit the arterial input function from the concentration series.

    Scores every brain voxel for arterial character (early, narrow, high peak),
    restricted to the vessel mask when it is large enough, then fits a
    gamma-variate to the best candidates.

    Returns a dict with keys 'gv' (fitted curve), 'time' and 'TR'.
    """
    coords, curves = _global_arterial_score(
        conc, mask, n_top=N_TOP_AIF, vessel_mask=vessels
    )

    if len(coords) == 0:
        raise ValueError("[AIF] No valid AIF candidates found in brain mask.")

    aif = _fit_aif_global(coords, curves, savepath, TR=FIXED_TR)
    aif["TR"] = FIXED_TR

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

import json
import numpy as np
from numpy import trapezoid
from scipy.ndimage import label as _scipy_label
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
# ==============================================================================
#                             MAIN AIF FUNCTION
# ==============================================================================

FIXED_TR = 1.5
N_TOP_AIF = 30
MAX_REFINEMENT_ITERS = 40
N_AVERAGE_CANDIDATES = 5   # voxels to average for the final AIF timecourse


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
            print(f"[AIF] Restricting AIF search to vessel mask ({n_vessel} voxels)")
            zz, rr, cc = np.where(vessel_in_mask)
        else:
            print(f"[AIF] Too few vessel-mask voxels ({n_vessel}) — using full brain")
            zz, rr, cc = np.where(mask)
    else:
        zz, rr, cc = np.where(mask)
    if len(zz) == 0:
        return [], np.zeros((0, nT))

    all_curves = conc[zz, rr, cc, :]          # (N, nT)
    peaks = all_curves.max(axis=1)
    ttp   = np.argmax(all_curves, axis=1).astype(float)

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
    print(f"[AIF] CTC peak median={peak_median:.1f}  arterial window=[{peak_lo:.1f}, {peak_hi:.1f}]")

    ttp_min = max(2, int(0.05 * nT))
    ttp_max = int(0.75 * nT)
    valid = (ttp >= ttp_min) & (ttp <= ttp_max) & (peaks >= peak_lo) & (peaks <= peak_hi)
    print(f"[AIF] Global search: {int(valid.sum())} / {len(zz)} brain voxels pass TTP+arterial window")

    if valid.sum() < n_top:
        # Relax lower bound — keep upper bound to still exclude bone
        print(f"[AIF] Too few candidates — relaxing to all peaks ≤ {peak_hi:.1f}")
        valid = (ttp >= ttp_min) & (ttp <= ttp_max) & (peaks > 0) & (peaks <= peak_hi)

    if valid.sum() < n_top:
        print("[AIF] Warning: few voxels pass TTP window — using all brain voxels")
        valid = peaks > 0

    v_zz = zz[valid]; v_rr = rr[valid]; v_cc = cc[valid]
    v_curves = all_curves[valid]
    v_peaks  = peaks[valid]
    v_ttp    = ttp[valid]

    widths = np.sum(v_curves > (v_peaks / 2.0)[:, None], axis=1).astype(float)
    widths = np.maximum(widths, 1.0)

    def _norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-9 else np.zeros_like(x)

    score = _norm01(v_peaks) - 0.4 * _norm01(v_ttp) - 0.4 * _norm01(widths)

    top_k = min(n_top, len(score))
    top_idx = np.argsort(score)[-top_k:]

    coords = list(zip(v_zz[top_idx].tolist(), v_rr[top_idx].tolist(), v_cc[top_idx].tolist()))
    curves = v_curves[top_idx]

    slices_used = sorted({int(v_zz[i]) for i in top_idx})
    print(f"[AIF] Global top-{top_k} candidates from slices: {slices_used}")
    return coords, curves


# ==============================================================================
# PYPET-STYLE CLUSTER SELECTION  (experiment/pypet-aif branch)
# Revert: change DSC_aif to call _global_arterial_score instead.
# ==============================================================================

def _cluster_arterial_score(conc, mask, vessel_mask=None):
    """
    PyPeT-style AIF candidate selection (Borghouts & Su, 2025):
      1. Score brain voxels by AUC (trapezoid integral) and TTP
      2. Apply progressively-relaxed percentile thresholds to find
         high-AUC, early-TTP candidates
      3. 3D connected-component analysis on candidates
      4. Score each cluster: size × mean_peak / fit_error
      5. Return voxels of the best-scoring cluster

    Falls back to top-N individual voxels (original approach) if no
    valid clusters are found.
    """
    nZ, nY, nX, nT = conc.shape

    if vessel_mask is not None:
        search_mask = mask & vessel_mask
        n_vessel = int(search_mask.sum())
        print(f"[AIF-cluster] Vessel mask: {n_vessel} voxels inside brain mask")
        if n_vessel < 10:
            print("[AIF-cluster] Too few vessel voxels — searching full brain")
            search_mask = mask
    else:
        search_mask = mask

    zz, rr, cc = np.where(search_mask)
    if len(zz) == 0:
        return [], np.zeros((0, nT))

    all_curves = conc[zz, rr, cc, :]          # (N, nT)
    auc        = trapezoid(all_curves, axis=1) # integral — PyPeT's primary metric
    ttp        = np.argmax(all_curves, axis=1).astype(float)

    auc_map = np.zeros((nZ, nY, nX))
    ttp_map = np.full((nZ, nY, nX), float(nT))
    auc_map[zz, rr, cc] = auc
    ttp_map[zz, rr, cc] = ttp

    pos_auc  = auc[auc > 0]
    valid_ttp = ttp[ttp < nT]
    if len(pos_auc) == 0:
        return [], np.zeros((0, nT))

    # Progressive thresholds: strict first, relax until ≥5 candidates found
    candidate_mask = None
    for auc_pct, ttp_pct in [(90, 20), (80, 30), (70, 40), (60, 50), (50, 60)]:
        auc_thr = float(np.percentile(pos_auc, auc_pct))
        ttp_thr = float(np.percentile(valid_ttp, ttp_pct)) if len(valid_ttp) > 0 else float(nT)
        cands = search_mask & (auc_map >= auc_thr) & (ttp_map <= ttp_thr)
        if int(cands.sum()) >= 5:
            candidate_mask = cands
            print(f"[AIF-cluster] AUC≥p{auc_pct} + TTP≤p{ttp_pct}: "
                  f"{int(cands.sum())} candidate voxels")
            break

    if candidate_mask is None:
        print("[AIF-cluster] No threshold satisfied — using all search voxels")
        candidate_mask = search_mask

    # 3D connected components (26-connectivity)
    struct = np.ones((3, 3, 3), dtype=np.int32)
    labeled, n_comp = _scipy_label(candidate_mask.astype(np.uint8), structure=struct)
    print(f"[AIF-cluster] {n_comp} connected components")

    MIN_CLUSTER, MAX_CLUSTER = 2, 3000
    time_vec = np.arange(nT) * FIXED_TR

    best_score  = -np.inf
    best_coords = None
    best_curves = None

    for comp_id in range(1, n_comp + 1):
        c_zz, c_rr, c_cc = np.where(labeled == comp_id)
        if len(c_zz) < MIN_CLUSTER or len(c_zz) > MAX_CLUSTER:
            continue

        cluster_curves = conc[c_zz, c_rr, c_cc, :]
        mean_ctc = cluster_curves.mean(axis=0)
        peak = float(mean_ctc.max())
        if peak <= 0:
            continue

        try:
            w = 0.01 + np.exp(-mean_ctc)
            TTP_idx = max(0, min(int(np.argmax(mean_ctc)), nT - 1))
            w[TTP_idx] /= 10.0
            params  = fitGV_peak1(mean_ctc.copy(), w.copy(), FIXED_TR, time_vec)
            gv      = GVfunction_peak1(params, FIXED_TR, time_vec)
            fit_err = float(np.mean((mean_ctc - gv) ** 2)) + 1e-6
        except Exception:
            fit_err = 1e6

        score = len(c_zz) * peak / fit_err
        if score > best_score:
            best_score  = score
            best_coords = list(zip(c_zz.tolist(), c_rr.tolist(), c_cc.tolist()))
            best_curves = cluster_curves

    if best_coords is None:
        print("[AIF-cluster] No valid clusters — falling back to top-AUC voxels")
        top_k   = min(N_TOP_AIF, len(zz))
        top_idx = np.argsort(auc)[-top_k:]
        best_coords = list(zip(zz[top_idx].tolist(), rr[top_idx].tolist(), cc[top_idx].tolist()))
        best_curves = all_curves[top_idx]
    else:
        slices_used = sorted({c[0] for c in best_coords})
        print(f"[AIF-cluster] Winning cluster: {len(best_coords)} voxels "
              f"from slices {slices_used}, score={best_score:.2f}")

    return best_coords, best_curves


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
        if TTP - 1 >= 0: weights[TTP - 1] /= 5.0
        if TTP + 1 < nT:  weights[TTP + 1] /= 2.0
        fit_params[i] = fitGV_peak1(curve.copy(), weights.copy(), TR, time)

    valid_ab = (fit_params[:, 1] > 0) & (fit_params[:, 2] > 0)
    if valid_ab.sum() > 0:
        avg_alpha = float(np.mean(fit_params[valid_ab, 1]))
        avg_beta  = float(np.mean(fit_params[valid_ab, 2]))
    else:
        avg_alpha, avg_beta = 5.0, 1.5
    print(f"[AIF] Initial avg_alpha={avg_alpha:.3f}, avg_beta={avg_beta:.3f}")

    fit_bat  = np.zeros((n, 2), dtype=float)
    fit_ab   = np.zeros((n, 2), dtype=float)
    fit_temp = fit_params.copy()

    print(f"[AIF] Starting refinement loop (max {MAX_REFINEMENT_ITERS} iterations)...")
    for nTrue in range(1, MAX_REFINEMENT_ITERS + 1):
        for i in range(n):
            curve = curves[i].astype(float)
            weights = 0.01 + np.exp(-curve)
            TTP = int(np.argmax(curve))
            TTP = max(0, min(TTP, nT - 1))
            weights[TTP] /= 10.0
            if TTP - 1 >= 0: weights[TTP - 1] /= 5.0
            if TTP + 1 < nT:  weights[TTP + 1] /= 2.0
            try:
                fit_bat[i] = find_params_bat_optim(
                    curve, weights, [avg_alpha, avg_beta],
                    fit_temp[i, 0], fit_temp[i, 3], TR, time,
                )
                fit_ab[i] = find_params_alpha_beta_optim(
                    curve, weights, fit_bat[i],
                    avg_alpha, avg_beta, TR, time,
                )
            except Exception:
                pass
            fit_temp[i, 0] = fit_bat[i, 0]
            fit_temp[i, 3] = fit_bat[i, 1]
            fit_temp[i, 1] = fit_ab[i, 0]
            fit_temp[i, 2] = fit_ab[i, 1]

        # Clamp to physiological ranges to prevent alternating-fit runaway
        fit_temp[:, 1] = np.clip(fit_temp[:, 1], 1.0, 15.0)   # alpha
        fit_temp[:, 2] = np.clip(fit_temp[:, 2], 0.3, 10.0)   # beta

        valid_ab = (fit_temp[:, 1] > 0) & (fit_temp[:, 2] > 0)
        if valid_ab.sum() > 0:
            avg_alpha = float(np.mean(fit_temp[valid_ab, 1]))
            avg_beta  = float(np.mean(fit_temp[valid_ab, 2]))

        if nTrue % 10 == 0:
            print(f"[AIF] refinement iter {nTrue}: avg_alpha={avg_alpha:.3f}, avg_beta={avg_beta:.3f}")

    print(f"[AIF] Reached MAX_REFINEMENT_ITERS={MAX_REFINEMENT_ITERS}, stopping.")

    # Compute per-voxel fit residual (MSE between measured curve and GV fit)
    residuals = np.zeros(n)
    for i in range(n):
        gv = GVfunction_peak1(fit_temp[i], TR, time)
        residuals[i] = float(np.mean((curves[i].astype(float) - gv) ** 2))

    # Keep voxels whose residual is below 2× the median — rejects badly-fit curves
    med_resid = float(np.median(residuals))
    good_fit = residuals < 2.0 * med_resid
    if good_fit.sum() == 0:
        good_fit = np.ones(n, dtype=bool)  # fallback: all
    print(f"[AIF] Fit quality: {int(good_fit.sum())}/{n} voxels pass residual gate "
          f"(median MSE={med_resid:.4f})")

    # Rank good-fit voxels by t0 (ascending) and average the top-N.
    # Normalising each curve to its GV peak before averaging removes amplitude
    # differences caused by partial volume; re-scaling by the median GV peak
    # restores a physically meaningful amplitude.  Re-fitting GV to the averaged
    # curve gives clean, low-noise parameters for deconvolution.
    t0_vals = fit_temp[:, 0].copy()
    t0_vals[~good_fit] = np.inf
    sorted_idx = np.argsort(t0_vals)

    n_avg = min(N_AVERAGE_CANDIDATES, int(good_fit.sum()))
    n_avg = max(n_avg, 1)
    avg_idx = sorted_idx[:n_avg]

    gv_peaks = np.array([
        max(float(GVfunction_peak1(fit_temp[i], TR, time).max()), 1.0)
        for i in avg_idx
    ])
    normalized = curves[avg_idx].astype(float) / gv_peaks[:, None]
    avg_conc   = normalized.mean(axis=0) * float(np.median(gv_peaks))

    avg_weights = 0.01 + np.exp(-avg_conc)
    TTP_avg = int(np.argmax(avg_conc)); TTP_avg = max(0, min(TTP_avg, nT - 1))
    avg_weights[TTP_avg] /= 10.0
    if TTP_avg - 1 >= 0: avg_weights[TTP_avg - 1] /= 5.0
    if TTP_avg + 1 < nT:  avg_weights[TTP_avg + 1] /= 2.0
    final_params = fitGV_peak1(avg_conc.copy(), avg_weights.copy(), TR, time)

    final_conc = avg_conc
    best_coord = coords[avg_idx[0]]   # earliest-t0 voxel used as representative coord

    avg_t0s = [float(fit_temp[i, 0]) for i in avg_idx]
    avg_coords = [coords[i] for i in avg_idx]
    print(f"[AIF] Averaging {n_avg} candidates: t0={[f'{v:.2f}' for v in avg_t0s]}s  "
          f"coords={avg_coords}")

    gv_curve = GVfunction_peak1(final_params, TR, time)
    print(f"[AIF] Averaged AIF: GV peak={float(gv_curve.max()):.1f} HU  "
          f"t0={float(final_params[0]):.2f}s  (rep. voxel {best_coord})")

    # Diagnostic: AIF curve + fit
    fig, ax = plt.subplots(figsize=(8, 4))
    for curve in curves:
        ax.plot(time, curve, color="steelblue", alpha=0.2, linewidth=0.8)
    for i in avg_idx:
        ax.plot(time, curves[i], color="orange", alpha=0.6, linewidth=1.0)
    ax.plot(time, final_conc, "bo", markersize=5, label=f"Averaged ({n_avg} voxels)")
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
        "t0_s":        float(final_params[0]),
        "alpha":       float(final_params[1]),
        "beta":        float(final_params[2]),
        "A":           float(final_params[3]),
        "peak_gv":     float(np.max(gv_curve)),
        "integral_gv": float(trapezoid(gv_curve, time)),
        "TR_s":        float(TR),
        "nT":          int(nT),
        "best_coord":  [int(x) for x in best_coord],
        "avg_coords":  [[int(x) for x in coords[i]] for i in avg_idx],
        "avg_t0s_s":   [float(fit_temp[i, 0]) for i in avg_idx],
        "n_averaged":  int(n_avg),
    }
    with open(f"{savepath}/aif_params.json", "w") as f:
        json.dump(params_dict, f, indent=2)

    print("[AIF] Final parameters:", params_dict)

    return {
        "parameters": final_params,
        "gv":         gv_curve,
        "time":       time,
        "conc":       final_conc,
    }


def DSC_aif(mask, conc, bolus, size, config, device, vessels, savepath):
    # EXPERIMENT (experiment/pypet-aif): using cluster-based selection.
    # To revert: swap _cluster_arterial_score → _global_arterial_score below.
    coords, curves = _cluster_arterial_score(conc, mask, vessel_mask=vessels)

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

    gv_for_A = GVfunction_peak1([t0_init, alpha_init, beta_init, 1.0], TR, time)
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
        obj, p0, bounds=(lb, ub),
        max_nfev=1000, xtol=1e-8, gtol=1e-8, ftol=1e-8,
    )
    return np.transpose(res.x)


def GVfunction_peak1(p, TR, time):
    t0, alpha, beta, A = p
    time = np.asarray(time, float)
    GV = np.zeros_like(time)
    mask = time > t0
    x = time[mask] - t0
    GV[mask] = A * (x ** alpha) * np.exp(-x / max(beta, 1e-6))
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
        obj, p0, bounds=(lb, ub),
        max_nfev=1000, xtol=1e-8, gtol=1e-8, ftol=1e-8,
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
    GV[mask] = A * (x ** alpha) * np.exp(-x / max(beta, 1e-6))
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
        obj, p0, bounds=(lb, ub),
        max_nfev=1000, xtol=1e-8, gtol=1e-8, ftol=1e-8,
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
    GV[mask] = A * (x ** alpha) * np.exp(-x / max(beta, 1e-6))
    return GV


def find_params_bat_optim(avg_test, avg_weights, fitParams_test, t0, A, TR, time):
    param1 = fitParams_test[0]
    param2 = fitParams_test[1]
    return fitGV_peak1_2(avg_test, avg_weights, param1, param2, t0, A, TR, time)[0:2]


def find_params_alpha_beta_optim(
    avg_test, avg_weights, fitParams_test, alpha, beta, TR, time
):
    param1 = fitParams_test[0]
    param2 = fitParams_test[1]
    return fitGV_peak1_1(
        avg_test, avg_weights, param1, param2, alpha, beta, TR, time,
    )[0:2]
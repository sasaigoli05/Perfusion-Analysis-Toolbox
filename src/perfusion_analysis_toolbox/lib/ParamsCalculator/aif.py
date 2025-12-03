import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from perfusion_analysis_toolbox.lib.ParamsCalculator.register import register_and_select_slice

# Optional: uniGradICON registration hook (disabled for now for debugging)
# from perfusion_analysis_toolbox.lib.register import register_and_select_slice

# ==============================================================================
#                         CONFIG CONSTANTS FOR SPEED
# ==============================================================================

# How many voxels in the ROI to actually fit (by highest peak signal)
N_TOP_AIF = 30

# Maximum number of refinement iterations
MAX_REFINEMENT_ITERS = 40


# ==============================================================================
#                             MAIN AIF FUNCTION
# ==============================================================================

def DSC_aif(
    mask,
    conc,
    bolus,
    size,
    config,
    device,
    vessels,
    savepath,
    cta_np=None,
    fixed_image_path=None,
):
    """
    Arterial Input Function (AIF) estimation.

    Refactored version that:
    - Preserves the original mathematical structure and workflow
    - Removes hard-coded nT = 16 assumption
    - Works with arbitrary time-length perfusion data
    - Keeps the uniGradICON registration hook commented for now
    """

    # ----------------------------------------------------------------------
    # GPU REGISTRATION (commented out for fast debugging)
    # ----------------------------------------------------------------------
    
    if cta_np is not None and fixed_image_path is not None:
        try:
            nSlice = register_and_select_slice(
                fixed_image_path, cta_np, fixed_slice_index=2
            )
            print(f"[AIF] Using GPU-mapped AIF slice = {nSlice}")
        except Exception as e:
            print("[AIF] Registration failed, falling back to default slice:", e)
            nSlice = 7
    else:
        nSlice = 7
    

    # DEBUG MODE: use the original fixed slice
    # nSlice = 7
    # print(f"[AIF] Warning: Using default AIF Z-slice = {nSlice}")

    # ----------------------------------------------------------------------
    # Slice data exactly as in the original implementation
    # ----------------------------------------------------------------------
    conc_reshaped = conc[nSlice - 1, :, :, :]
    mask_slice = mask[nSlice - 1, :, :]
    vessel_slice = vessels[nSlice - 1, :, :]
    bolus_slice = bolus[nSlice - 1]

    # TR is fixed by acquisition; keep original value
    TR = 1.55

    print(f"[AIF] slice shape: {conc_reshaped.shape}, TR={TR}")
    aif = extract_AIF(
        conc_reshaped,
        mask_slice,
        vessel_slice,
        bolus_slice,
        savepath,
        TR=TR,
    )
    return aif


# ==============================================================================
#                              AIF EXTRACTION
# ==============================================================================

def extract_AIF(AIFslice, mask, vesselSlice, bolusSlice, savepath, TR=1.55):
    """
    Extract the AIF from a 2D slice of the 4D concentration image.

    Refactoring details:
    - nT is taken from AIFslice.shape[-1] (no fixed 16)
    - TR/time passed consistently into all fitting helpers
    - Bounds and indexing are guarded to avoid previous errors
    - SPEED: only top N_TOP_AIF voxels (by peak signal) in ROI are used
             and refinement iterations are capped at MAX_REFINEMENT_ITERS.
    """

    semiaxisMag = 0.2500
    semiaxisMin = 0.1000

    nR, nC, nT = AIFslice.shape
    time = np.arange(nT) * TR

    print("[AIF] Brain bound detection")

    # ------------------------------------------------------------------
    # Mask bounding box (same logic, with safe loop limits)
    # ------------------------------------------------------------------
    r = 0
    while r < nR:
        if np.sum(mask[r, :]) != 0:
            minR = r
            break
        r += 1

    r = nR - 1
    while r >= 0:
        if np.sum(mask[r, :]) != 0:
            maxR = r
            break
        r -= 1

    c = 0
    while c < nC:
        if np.sum(mask[:, c]) != 0:
            minC = c
            break
        c += 1

    c = nC - 1
    while c >= 0:
        if np.sum(mask[:, c]) != 0:
            maxC = c
            break
        c -= 1

    print("[AIF] Definition of the AIF extraction searching area")

    center = np.zeros(2, dtype=float)
    center[1] = 0.5 * (minR + maxR)  # Y coordinate
    center[0] = 0.6 * (minC + maxC)  # X coordinate

    semiaxisB = semiaxisMag * (maxC - minC)
    semiaxisA = semiaxisMin * (maxR - minR)

    ROI = np.zeros((nR, nC), dtype=float)
    count = 0
    for r in range(nR):
        for c in range(nC):
            val = (
                ((r - center[1]) ** 2) / (semiaxisA ** 2)
                + ((c - center[0]) ** 2) / (semiaxisB ** 2)
            )
            if val <= 1:
                ROI[r, c] = 1.0
                count += 1

    ROI = ROI * mask

    # vessel filter (same behavior, but vectorized)
    ROI_filtered = (vesselSlice == 1) & (ROI == 1)
    ROI = ROI_filtered.astype(float) * mask

    if not np.any(ROI):
        raise ValueError("AIF ROI is empty.")

    # ------------------------------------------------------------------
    # SPEED OPTIMIZATION: only keep top N_TOP_AIF voxels in ROI
    # ------------------------------------------------------------------
    ind = np.where(ROI)  # candidate voxels
    num_candidates = len(ind[0])
    print(f"[AIF] ROI candidate voxels: {num_candidates}")

    # Extract peak concentration for each candidate voxel
    voxel_curves = AIFslice[ind[0], ind[1], :]  # shape (N, nT)
    peaks = voxel_curves.max(axis=1)

    top_k = min(N_TOP_AIF, num_candidates)
    if top_k <= 0:
        raise ValueError("No valid voxels found in AIF ROI.")

    # indices of top_k voxels by peak value
    top_idx = np.argsort(peaks)[-top_k:]
    print(f"[AIF] Using top {top_k} voxels for AIF fitting")

    # build maskAIF that only includes selected voxels
    maskAIF = np.zeros_like(ROI)
    for idx in top_idx:
        r = ind[0][idx]
        c = ind[1][idx]
        maskAIF[r, c] = 1.0

    # ------------------------------------------------------------------
    # For completeness: data2D (not strictly required downstream)
    # ------------------------------------------------------------------
    sel_ind = np.where(maskAIF)
    data2D = np.zeros((len(sel_ind[0]), nT), dtype=float)
    for t in range(nT):
        data2D[:, t] = AIFslice[sel_ind[0], sel_ind[1], t]

    fit_Params = np.zeros((nR, nC, 4), dtype=float)

    # ------------------------------------------------------------------
    # FIRST PASS FIT (over selected voxels only)
    # ------------------------------------------------------------------
    count = 0
    print("[AIF] Starting first-pass GV fits...")
    for r in range(nR):
        for c in range(nC):
            if maskAIF[r, c] == 1:
                count += 1
                AIFconc = AIFslice[r, c, :].astype(float)
                weights = 0.01 + np.exp(-AIFconc)

                TTP = int(np.argmax(AIFconc))
                if 0 <= TTP < nT:
                    weights[TTP] /= 10.0
                if TTP - 1 >= 0:
                    weights[TTP - 1] /= 5.0
                if TTP + 1 < nT:
                    weights[TTP + 1] /= 2.0

                fitParams = fitGV_peak1(AIFconc.copy(), weights.copy(), TR, time)
                fit_Params[r, c, :] = fitParams[0:4]

    print(f"[AIF] First-pass fits done for {count} voxels")
    avg_alpha, avg_beta = find_avg_alpha_beta(maskAIF, fit_Params, count, nR, nC)
    print(f"[AIF] Initial avg_alpha={avg_alpha:.3f}, avg_beta={avg_beta:.3f}")

    # ------------------------------------------------------------------
    # ITERATIVE REFINEMENT (same logic, capped at MAX_REFINEMENT_ITERS)
    # ------------------------------------------------------------------
    nTrue = 0
    fitParams_bat = np.zeros((nR, nC, 2), dtype=float)
    fitParams_ab = np.zeros((nR, nC, 2), dtype=float)
    fit_Params_temp = np.zeros((nR, nC, 4), dtype=float)

    print(f"[AIF] Starting refinement loop (max {MAX_REFINEMENT_ITERS} iterations)...")
    while True:
        nTrue += 1
        for r in range(nR):
            for c in range(nC):
                if maskAIF[r, c] == 1:
                    AIFconc = AIFslice[r, c, :].astype(float)
                    weights = 0.01 + np.exp(-AIFconc)
                    TTP = int(np.argmax(AIFconc))
                    if 0 <= TTP < nT:
                        weights[TTP] /= 10.0
                    if TTP - 1 >= 0:
                        weights[TTP - 1] /= 5.0
                    if TTP + 1 < nT:
                        weights[TTP + 1] /= 2.0

                    fitParams_bat[r, c, :] = find_params_bat_optim(
                        AIFconc,
                        weights,
                        [avg_alpha, avg_beta],
                        fit_Params_temp[r, c, 0],
                        fit_Params_temp[r, c, 3],
                        TR,
                        time,
                    )

                    fitParams_ab[r, c, :] = find_params_alpha_beta_optim(
                        AIFconc,
                        weights,
                        fitParams_bat[r, c, :],
                        avg_alpha,
                        avg_beta,
                        TR,
                        time,
                    )

                    fit_Params_temp[r, c, 0] = fitParams_bat[r, c, 0]
                    fit_Params_temp[r, c, 3] = fitParams_bat[r, c, 1]
                    fit_Params_temp[r, c, 1] = fitParams_ab[r, c, 0]
                    fit_Params_temp[r, c, 2] = fitParams_ab[r, c, 1]

        avg_alpha, avg_beta = find_avg_alpha_beta(
            maskAIF, fit_Params_temp, count, nR, nC
        )

        if nTrue % 10 == 0:
            print(
                f"[AIF]  refinement iter {nTrue}: avg_alpha={avg_alpha:.3f}, "
                f"avg_beta={avg_beta:.3f}"
            )

        if nTrue >= MAX_REFINEMENT_ITERS:
            print(f"[AIF] Reached MAX_REFINEMENT_ITERS={MAX_REFINEMENT_ITERS}, stopping.")
            break

    # ------------------------------------------------------------------
    # FINAL SELECTION: MINIMAL t0
    # ------------------------------------------------------------------
    min_t0 = np.inf
    final_fitParameters = np.zeros(4, dtype=float)
    final_conc = np.zeros(nT, dtype=float)

    for r in range(nR):
        for c in range(nC):
            if maskAIF[r, c] == 1:
                t0 = fit_Params_temp[r, c, 0]
                if t0 < min_t0:
                    min_t0 = t0
                    final_fitParameters = fit_Params_temp[r, c, :].copy()
                    final_conc = AIFslice[r, c, :].astype(float)

    # ------------------------------------------------------------------
    # FINALIZE OUTPUT
    # ------------------------------------------------------------------
    AIF_fit = {
        "parameters": final_fitParameters,
        "gv": GVfunction_peak1(final_fitParameters, TR, time),
        "time": time,
        "conc": final_conc,
    }

    plt.figure()
    plt.plot(time, AIF_fit["conc"], "bo", markersize=5)
    plt.plot(time, AIF_fit["gv"], "k-", linewidth=2)
    plt.legend(["AIF samples", "GV function"])
    plt.title("AIF")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(f"{savepath}/high-res-aif.png")
    plt.close()

    print("[AIF] Final AIF parameters:", final_fitParameters)
    return AIF_fit


# ==============================================================================
#                   FITTING HELPER FUNCTIONS (TIME-AWARE)
# ==============================================================================

def sanitize_bounds(p0):
    """
    Ensure valid bounds for least_squares:
    - p0 strictly positive
    - lower < upper component-wise
    """
    p0 = np.asarray(p0, float)
    p0 = np.maximum(p0, 1e-6)

    lb = 0.1 * p0
    ub = 10.0 * p0
    too_close = ub - lb <= 1e-8
    ub[too_close] = lb[too_close] + 1e-6

    return lb, ub


def make_feasible_initial_guess(p0):
    """
    Given an initial p0, build bounds and then clamp p0 so that
    lb < p0 < ub for every dimension. This guarantees that
    least_squares won't complain about 'initial guess outside bounds'.
    """
    p0 = np.asarray(p0, float)
    lb, ub = sanitize_bounds(p0)

    # clamp p0 strictly inside [lb, ub]
    eps = 1e-6
    p0 = np.maximum(p0, lb + eps)
    p0 = np.minimum(p0, ub - eps)

    return p0, lb, ub


def fitGV_peak1(data, weights, TR, time):
    """
    Fit the first peak with a gamma-variate function.
    Parameters: p = [t0, alpha, beta, A]
    """
    data = np.asarray(data, float)
    weights = np.asarray(weights, float)
    nT = data.shape[0]

    MCdata = np.max(data)
    TTPpos = int(np.argmax(data))
    TTPpos = max(0, min(TTPpos, nT - 1))

    if TTPpos > 0:
        idx = np.where(data[:TTPpos] <= 0.05 * MCdata)[0]
        if idx.size > 0:
            last_idx = idx[-1]
        else:
            last_idx = 0
    else:
        last_idx = 0

    t0_init = time[last_idx]
    alpha_init = 5.0
    TTPdata = time[TTPpos]
    beta_init = max((TTPdata - t0_init) / max(alpha_init, 1e-6), 1e-3)

    gv_for_A = GVfunction_peak1([t0_init, alpha_init, beta_init, 1.0], TR, time)
    max_gv = np.max(gv_for_A)
    if max_gv <= 0:
        A_init = MCdata
    else:
        A_init = MCdata / max_gv

    p0 = np.array([t0_init, alpha_init, beta_init, A_init], dtype=float)

    # enforce bounds + feasible x0
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
    """
    Gamma-variate function for peak 1:
    GV(t) = A * ((t - t0)**alpha) * exp(-(t - t0)/beta)
    """
    t0, alpha, beta, A = p
    time = np.asarray(time, float)
    GV = np.zeros_like(time)
    mask = time > t0
    x = time[mask] - t0
    GV[mask] = A * (x ** alpha) * np.exp(-x / max(beta, 1e-6))
    return GV


def fitGV_peak1_2(data, weights, param1, param2, t0, A, TR, time):
    """
    Fit t0 and A, given fixed alpha=param1, beta=param2.
    """
    data = np.asarray(data, float)
    weights = np.asarray(weights, float)
    nT = data.shape[0]

    MCdata = np.max(data)
    TTPpos = int(np.argmax(data))
    TTPpos = max(0, min(TTPpos, nT - 1))

    if TTPpos > 0:
        idx = np.where(data[:TTPpos] <= 0.05 * MCdata)[0]
        if idx.size > 0:
            last_idx = idx[-1]
        else:
            last_idx = 0
    else:
        last_idx = 0

    if t0 == 0:
        t0_init = time[last_idx]
    else:
        t0_init = t0

    if A == 0:
        gv_for_A = GVfunction_peak1_2([t0_init, 1.0], param1, param2, TR, time)
        max_gv = np.max(gv_for_A)
        if max_gv <= 0:
            A_init = MCdata
        else:
            A_init = MCdata / max_gv
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
    """
    Gamma-variate with fixed alpha=param1, beta=param2.
    p = [t0, A]
    """
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
    """
    Fit [alpha, beta] given fixed t0=param1 and A=param2.
    """
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
    """
    Gamma-variate with t0=param1, A=param2.
    p = [alpha, beta]
    """
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
    """
    Wrapper around fitGV_peak1_2 to update [t0, A].
    fitParams_test = [alpha_avg, beta_avg]
    """
    param1 = fitParams_test[0]
    param2 = fitParams_test[1]
    return fitGV_peak1_2(avg_test, avg_weights, param1, param2, t0, A, TR, time)[0:2]


def find_avg_alpha_beta(maskAIF, fit_Params, count, nR, nC):
    """
    Compute average alpha and beta over the AIF mask.
    """
    alpha_sum = 0.0
    beta_sum = 0.0
    for r in range(nR):
        for c in range(nC):
            if maskAIF[r, c] == 1:
                alpha_sum += fit_Params[r, c, 1]
                beta_sum += fit_Params[r, c, 2]
    if count == 0:
        return 0.0, 0.0
    return alpha_sum / count, beta_sum / count


def find_params_alpha_beta_optim(
    avg_test, avg_weights, fitParams_test, alpha, beta, TR, time
):
    """
    Wrapper around fitGV_peak1_1 to update [alpha, beta].
    fitParams_test = [t0, A] from previous step.
    """
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

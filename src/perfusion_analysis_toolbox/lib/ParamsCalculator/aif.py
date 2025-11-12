# # aif.py
# import numpy as np
# from scipy.optimize import least_squares
# import matplotlib.pyplot as plt


# # ----------------------------
# # Shape helpers
# # ----------------------------
# def _center_fit_2d(arr, target_shape):
#     """Center-crop or zero-pad a 2D array to target_shape=(rows, cols)."""
#     r0, c0 = arr.shape
#     rt, ct = target_shape
#     out = np.zeros((rt, ct), dtype=arr.dtype)

#     ro = min(r0, rt)
#     co = min(c0, ct)

#     # source window (centered)
#     rs = (r0 - ro) // 2
#     cs = (c0 - co) // 2
#     re = rs + ro
#     ce = cs + co

#     # target window (centered)
#     rts = (rt - ro) // 2
#     cts = (ct - co) // 2
#     rte = rts + ro
#     cte = cts + co

#     out[rts:rte, cts:cte] = arr[rs:re, cs:ce]
#     return out


# def _ensure_slice_shapes_match(mask_slice, vessel_slice, aif_shape_rc):
#     """Make mask/vessel slices match AIF (R,C) by center crop/pad."""
#     r, c = aif_shape_rc
#     if mask_slice.shape != (r, c):
#         mask_slice = _center_fit_2d(mask_slice, (r, c))
#     if vessel_slice is not None and vessel_slice.shape != (r, c):
#         vessel_slice = _center_fit_2d(vessel_slice, (r, c))
#     return mask_slice, vessel_slice


# def _safe_time_vector(nT, TR, proposed_time=None):
#     """
#     Always return a time vector of length nT.
#     If proposed_time is provided but length!=nT, rebuild from TR.
#     """
#     if proposed_time is None:
#         return np.arange(0, nT * TR, TR, dtype=float)
#     t = np.asarray(proposed_time, float).reshape(-1)
#     if t.size != nT:
#         # Rebuild to match samples (prevents matplotlib x/y mismatch).
#         return np.linspace(0.0, (nT - 1) * TR, num=nT, dtype=float)
#     return t


# # ----------------------------
# # Public entry point
# # ----------------------------
# def DSC_aif(self, mask, conc, bolus, size, config, device, vessels, TR=1.55):
#     """
#     Inputs
#     ------
#     mask    : (Z,R,C) uint8/bool brain mask
#     conc    : (Z,R,C,T) float perfusion / concentration series
#     bolus   : (Z,) or (Z,?) optional, not used here
#     vessels : (Z,R,C) uint8/bool candidate vessel mask (can be all zeros)
#     TR      : seconds per frame

#     Returns
#     -------
#     AIF dict with keys:
#       "parameters" [t0, alpha, beta, A],
#       "gv" (fit curve),
#       "time" (vector),
#       "conc" (raw AIF voxel curve)
#     """

#     # ---- Harmonize Z (use min depth across inputs)
#     Zc = conc.shape[0]
#     Zm = mask.shape[0]
#     Zv = vessels.shape[0] if vessels is not None else Zm
#     Z = min(Zc, Zm, Zv)

#     concZ = conc[:Z, ...]
#     maskZ = mask[:Z, ...]
#     vesselsZ = vessels[:Z, ...] if vessels is not None else np.zeros_like(maskZ, dtype=bool)

#     # ---- Choose axial slice: prefer mask∩vessels, else largest mask area
#     z_sums = (vesselsZ.astype(bool) & maskZ.astype(bool)).sum(axis=(1, 2))
#     if np.any(z_sums > 0):
#         nSlice = int(np.argmax(z_sums))
#     else:
#         nSlice = int(np.argmax(maskZ.reshape(maskZ.shape[0], -1).sum(axis=1)))

#     # ---- Extract slices
#     AIFslice    = concZ[nSlice, :, :, :]                   # (R,C,T)
#     maskSlice   = maskZ[nSlice, :, :].astype(bool)
#     vesselSlice = (vesselsZ[nSlice, :, :].astype(bool)
#                    if vessels is not None else np.zeros_like(maskSlice, bool))
#     bolusSlice  = (bolus[nSlice]
#                    if (bolus is not None and np.ndim(bolus) > 0 and nSlice < np.shape(bolus)[0])
#                    else None)

#     # ---- Ensure mask/vessel have same (R,C) as AIF slice
#     maskSlice, vesselSlice = _ensure_slice_shapes_match(maskSlice, vesselSlice, AIFslice.shape[:2])

#     # ---- Run extraction
#     return extract_AIF(AIFslice, maskSlice, vesselSlice, bolusSlice, TR=TR, save_fig="high-res-aif.png")


# # ----------------------------
# # Core logic (robust GV fit)
# # ----------------------------
# def extract_AIF(AIFslice, mask, vesselSlice, bolusSlice=None, TR=1.55, save_fig=None):
#     """
#     AIFslice : (R,C,T)
#     mask     : (R,C) bool
#     vesselSlice : (R,C) bool
#     """
#     nR, nC, nT = AIFslice.shape
#     time = _safe_time_vector(nT, TR)  # always length nT

#     # Safety: if called directly, fix mask/vessel shape again
#     if mask.shape != (nR, nC) or (vesselSlice is not None and vesselSlice.shape != (nR, nC)):
#         mask, vesselSlice = _ensure_slice_shapes_match(mask, vesselSlice, (nR, nC))

#     # -----------------------
#     # local helpers (closures)
#     # -----------------------
#     def _weights(curve):
#         y = np.asarray(curve, float)
#         T = y.size
#         w = 0.01 + np.exp(-np.clip(y, -1e6, 1e6))
#         ttp = int(np.argmax(y))
#         if 0 <= ttp < T:   w[ttp]      = max(w[ttp] / 10.0, 1e-6)
#         if 0 <= ttp-1 < T: w[ttp - 1]  = max(w[ttp - 1] / 5.0,  1e-6)
#         if 0 <= ttp+1 < T: w[ttp + 1]  = max(w[ttp + 1] / 2.0,  1e-6)
#         return w

#     def _gv(p, t):
#         t0, alpha, beta, A = [float(v) for v in p]
#         t = np.asarray(t, float)
#         y = np.zeros_like(t, float)
#         m = t > t0
#         x = np.maximum(t[m] - t0, 1e-12)
#         y[m] = A * (x ** alpha) * np.exp(-x / (beta + 1e-12))
#         return y

#     def _fit_gv_peak1(y, w, t):
#         y = np.asarray(y, float); w = np.asarray(w, float); t = np.asarray(t, float)
#         T = y.size; MC = float(np.max(y)); ttp_idx = int(np.argmax(y)); TTP = float(t[ttp_idx])

#         pre = np.where(y[:ttp_idx] <= 0.05 * MC)[0]
#         if pre.size:
#             t0_init = float(t[int(pre[-1])])
#         else:
#             base_end = max(1, T // 3)
#             base_mean = float(np.mean(y[:base_end])); base_std  = float(np.std(y[:base_end]) + 1e-6)
#             pre2 = np.where(y[:ttp_idx] <= base_mean + 2.0 * base_std)[0]
#             t0_init = float(t[int(pre2[-1])]) if pre2.size else float(t[0])

#         # end of first-pass fit region
#         i = ttp_idx
#         while i < T and y[i] > (0.2 * MC): i += 1
#         end = min(max(ttp_idx + 1, i), T)

#         t_fit = t[:end]; y_fit = y[:end]; w_fit = w[:end] if w.shape == y.shape else np.ones_like(y_fit)

#         y_tmp = _gv([t0_init, 5.0, max((TTP - t0_init) / (5.0 + 1e-6), 0.25), 1.0], t_fit)
#         y_tmp_max = float(np.max(y_tmp)) if np.any(y_tmp > 0) else 1e-6
#         A_init = max(MC / y_tmp_max, 1e-6)
#         p0 = np.array([t0_init, 5.0, max((TTP - t0_init) / (5.0 + 1e-6), 0.25), A_init], float)
#         lb = np.array([t[0] - (t[1]-t[0] if t.size > 1 else 0.5), 0.5, 0.1, 0.0], float)
#         ub = np.array([TTP, 20.0, 20.0, np.inf], float)

#         def resid(p): return (_gv(p, t_fit) - y_fit) / (w_fit + 1e-6)

#         res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=1000, xtol=1e-8, gtol=1e-8, ftol=1e-8)
#         return np.asarray(res.x, float)

#     def _fit_bat_and_A(y, w, alpha, beta, t0_hint, A_hint, t):
#         # solve for [t0, A] given fixed (alpha, beta)
#         y = np.asarray(y, float); w = np.asarray(w, float); t = np.asarray(t, float)
#         T = y.size; MC = float(np.max(y)); ttp_idx = int(np.argmax(y))
#         i = ttp_idx
#         while i < T and y[i] > (0.2 * MC): i += 1
#         end = min(max(ttp_idx + 1, i), T)
#         t_fit = t[:end]; y_fit = y[:end]; w_fit = w[:end] if w.shape == y.shape else np.ones_like(y_fit)

#         # init t0
#         pre = np.where(y[:ttp_idx] <= 0.05 * MC)[0]
#         if pre.size: t0_init = float(t[int(pre[-1])])
#         else:
#             base_end = max(1, T // 3)
#             base_mean = float(np.mean(y[:base_end])); base_std = float(np.std(y[:base_end]) + 1e-6)
#             pre2 = np.where(y[:ttp_idx] <= base_mean + 2.0 * base_std)[0]
#             t0_init = float(t[int(pre2[-1])]) if pre2.size else float(t[0])
#         if t0_hint is not None and np.isfinite(t0_hint): t0_init = float(t0_hint)

#         # init A
#         if A_hint is not None and np.isfinite(A_hint):
#             A_init = float(A_hint)
#         else:
#             tmp = _gv([t0_init, alpha, beta, 1.0], t_fit)
#             tmp_max = float(np.max(tmp)) if np.any(tmp > 0) else 1e-6
#             A_init = max(float(np.max(y_fit)) / tmp_max, 1e-6)

#         p0 = np.array([t0_init, A_init], float)
#         lb = np.array([t[0] - (t[1]-t[0] if t.size > 1 else 0.5), 0.0], float)
#         ub = np.array([t[ttp_idx], np.inf], float)

#         def resid(p): return (_gv([p[0], alpha, beta, p[1]], t_fit) - y_fit) / (w_fit + 1e-6)

#         res = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-8, gtol=1e-8, ftol=1e-8, max_nfev=1000)
#         return np.asarray(res.x, float)  # t0, A

#     def _fit_alpha_beta(y, w, t0, A, alpha_hint, beta_hint, t):
#         # solve for [alpha, beta] given fixed (t0, A)
#         y = np.asarray(y, float); w = np.asarray(w, float); t = np.asarray(t, float)
#         T = y.size; ttp_idx = int(np.argmax(y)); MC = float(np.max(y))
#         i = ttp_idx
#         while i < T and y[i] > (0.2 * MC): i += 1
#         end = min(max(ttp_idx + 1, i), T)
#         t_fit = t[:end]; y_fit = y[:end]; w_fit = w[:end] if w.shape == y.shape else np.ones_like(y_fit)

#         alpha_init = float(alpha_hint) if alpha_hint else 5.0
#         if beta_hint:
#             beta_init = float(beta_hint)
#         else:
#             pre = np.where(y[:ttp_idx] <= 0.05 * MC)[0]
#             if pre.size:
#                 t0_est = float(t[int(pre[-1])])
#             else:
#                 base_end = max(1, T // 3)
#                 base_mean = float(np.mean(y[:base_end])); base_std  = float(np.std(y[:base_end]) + 1e-6)
#                 pre2 = np.where(y[:ttp_idx] <= base_mean + 2.0 * base_std)[0]
#                 t0_est = float(t[int(pre2[-1])]) if pre2.size else float(t[0])
#             beta_init = max((float(t[ttp_idx]) - t0_est) / (alpha_init + 1e-6), 0.25)

#         p0 = np.array([alpha_init, beta_init], float)
#         lb = np.array([0.5, 0.1], float); ub = np.array([20.0, 20.0], float)

#         def resid(p): return (_gv([t0, p[0], p[1], A], t_fit) - y_fit) / (w_fit + 1e-6)

#         res = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-8, gtol=1e-8, ftol=1e-8, max_nfev=1000)
#         return np.asarray(res.x, float)  # alpha, beta

#     # -----------------------
#     # ROI and per-voxel fits
#     # -----------------------
#     rows_nonzero = np.where(mask.sum(axis=1) > 0)[0]
#     cols_nonzero = np.where(mask.sum(axis=0) > 0)[0]
#     if rows_nonzero.size == 0 or cols_nonzero.size == 0:
#         raise ValueError("Mask slice is empty; cannot compute AIF.")

#     ROI = (mask & vesselSlice) if (vesselSlice is not None and vesselSlice.any()) else mask
#     if ROI.sum() == 0:
#         raise ValueError("ROI is empty (mask∩vessels is zero).")

#     rr, cc = np.where(ROI)
#     fit_Params = np.zeros((nR, nC, 4), float)  # [t0, alpha, beta, A]

#     for r, c in zip(rr, cc):
#         y = AIFslice[r, c, :].astype(float)
#         p = _fit_gv_peak1(y, _weights(y), time)
#         fit_Params[r, c, :] = p

#     # Iteratively refine by alternating (t0,A) and (alpha,beta), update population means
#     tmp = np.zeros_like(fit_Params)

#     def _avg_ab(M, maskAIF):
#         idx = np.where(maskAIF)
#         return float(np.mean(M[idx[0], idx[1], 1])), float(np.mean(M[idx[0], idx[1], 2]))

#     avg_alpha, avg_beta = _avg_ab(fit_Params, ROI)
#     for _ in range(20):
#         for r, c in zip(rr, cc):
#             y = AIFslice[r, c, :].astype(float)
#             w = _weights(y)
#             t0, A = _fit_bat_and_A(y, w, avg_alpha, avg_beta, tmp[r, c, 0], tmp[r, c, 3], time)
#             alpha, beta = _fit_alpha_beta(y, w, t0, A, avg_alpha, avg_beta, time)
#             tmp[r, c, :] = [t0, alpha, beta, A]
#         avg_alpha, avg_beta = _avg_ab(tmp, ROI)

#     # Choose voxel with minimal t0 as final AIF
#     best = None
#     best_t0 = np.inf
#     for r, c in zip(rr, cc):
#         t0 = tmp[r, c, 0]
#         if np.isfinite(t0) and t0 < best_t0:
#             best_t0 = t0
#             best = (r, c)

#     if best is None:
#         raise RuntimeError("Failed to identify AIF voxel.")

#     r, c = best
#     p = tmp[r, c, :].copy()
#     aif_fit = {
#         "parameters": p,
#         "gv": _gv(p, time),
#         "time": time,
#         "conc": AIFslice[r, c, :].astype(float),
#     }

#     # Optional figure (time and conc are guaranteed same length)
#     if save_fig:
#         plt.figure()
#         plt.plot(time, aif_fit["conc"], "bo", markersize=4, label="AIF samples")
#         plt.plot(time, aif_fit["gv"], "k-", linewidth=2, label="GV fit")
#         plt.xlabel("Time [s]")
#         plt.legend()
#         plt.title("AIF")
#         plt.xlim(time[0], time[-1])
#         plt.tight_layout()
#         plt.savefig(save_fig, dpi=150)
#         plt.close()

#     return aif_fit
# aif.py
import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


# ----------------------------
# Shape helpers
# ----------------------------
def _center_fit_2d(arr, target_shape):
    """Center-crop or zero-pad a 2D array to target_shape=(rows, cols)."""
    r0, c0 = arr.shape
    rt, ct = target_shape
    out = np.zeros((rt, ct), dtype=arr.dtype)

    ro = min(r0, rt)
    co = min(c0, ct)

    # source window (centered)
    rs = (r0 - ro) // 2
    cs = (c0 - co) // 2
    re = rs + ro
    ce = cs + co

    # target window (centered)
    rts = (rt - ro) // 2
    cts = (ct - co) // 2
    rte = rts + ro
    cte = cts + co

    out[rts:rte, cts:cte] = arr[rs:re, cs:ce]
    return out


def _ensure_slice_shapes_match(mask_slice, vessel_slice, aif_shape_rc):
    """Make mask/vessel slices match AIF (R,C) by center crop/pad."""
    r, c = aif_shape_rc
    if mask_slice.shape != (r, c):
        mask_slice = _center_fit_2d(mask_slice, (r, c))
    if vessel_slice is not None and vessel_slice.shape != (r, c):
        vessel_slice = _center_fit_2d(vessel_slice, (r, c))
    return mask_slice, vessel_slice


def _safe_time_vector(nT, TR, proposed_time=None):
    """
    Always return a time vector of length nT.
    If proposed_time is provided but length!=nT, rebuild from TR.

    We use linspace to avoid floating-point endpoint drift that causes
    off-by-one length errors with np.arange for non-integer TR.
    """
    if proposed_time is None:
        # Guarantee exactly nT samples: [0, (nT-1)*TR]
        return np.linspace(0.0, (nT - 1) * TR, num=nT, dtype=float)

    t = np.asarray(proposed_time, float).reshape(-1)
    if t.size != nT:
        # Rebuild to match samples (prevents matplotlib x/y mismatch).
        return np.linspace(0.0, (nT - 1) * TR, num=nT, dtype=float)
    return t


# ----------------------------
# Public entry point
# ----------------------------
def DSC_aif(self, mask, conc, bolus, size, config, device, vessels, TR=1.55):
    """
    Inputs
    ------
    mask    : (Z,R,C) uint8/bool brain mask
    conc    : (Z,R,C,T) float perfusion / concentration series
    bolus   : (Z,) or (Z,?) optional, not used here
    vessels : (Z,R,C) uint8/bool candidate vessel mask (can be all zeros)
    TR      : seconds per frame

    Returns
    -------
    AIF dict with keys:
      "parameters" [t0, alpha, beta, A],
      "gv" (fit curve),
      "time" (vector),
      "conc" (raw AIF voxel curve)
    """

    # ---- Harmonize Z (use min depth across inputs)
    Zc = conc.shape[0]
    Zm = mask.shape[0]
    Zv = vessels.shape[0] if vessels is not None else Zm
    Z = min(Zc, Zm, Zv)

    concZ = conc[:Z, ...]
    maskZ = mask[:Z, ...]
    vesselsZ = vessels[:Z, ...] if vessels is not None else np.zeros_like(maskZ, dtype=bool)

    # ---- Choose axial slice: prefer mask∩vessels, else largest mask area
    z_sums = (vesselsZ.astype(bool) & maskZ.astype(bool)).sum(axis=(1, 2))
    if np.any(z_sums > 0):
        nSlice = int(np.argmax(z_sums))
    else:
        nSlice = int(np.argmax(maskZ.reshape(maskZ.shape[0], -1).sum(axis=1)))

    # ---- Extract slices
    AIFslice    = concZ[nSlice, :, :, :]                   # (R,C,T)
    maskSlice   = maskZ[nSlice, :, :].astype(bool)
    vesselSlice = (vesselsZ[nSlice, :, :].astype(bool)
                   if vessels is not None else np.zeros_like(maskSlice, bool))
    bolusSlice  = (bolus[nSlice]
                   if (bolus is not None and np.ndim(bolus) > 0 and nSlice < np.shape(bolus)[0])
                   else None)

    # ---- Ensure mask/vessel have same (R,C) as AIF slice
    maskSlice, vesselSlice = _ensure_slice_shapes_match(maskSlice, vesselSlice, AIFslice.shape[:2])

    # ---- Where to save the debug figure
    save_dir = getattr(self, "savepath", ".")
    save_fig = os.path.join(save_dir, "high-res-aif.png")

    # ---- Run extraction
    return extract_AIF(AIFslice, maskSlice, vesselSlice, bolusSlice, TR=TR, save_fig=save_fig)


# ----------------------------
# Core logic (robust GV fit)
# ----------------------------
def extract_AIF(AIFslice, mask, vesselSlice, bolusSlice=None, TR=1.55, save_fig=None):
    """
    AIFslice : (R,C,T)
    mask     : (R,C) bool
    vesselSlice : (R,C) bool
    """
    nR, nC, nT = AIFslice.shape
    time = _safe_time_vector(nT, TR)  # always length nT

    # Safety: if called directly, fix mask/vessel shape again
    if mask.shape != (nR, nC) or (vesselSlice is not None and vesselSlice.shape != (nR, nC)):
        mask, vesselSlice = _ensure_slice_shapes_match(mask, vesselSlice, (nR, nC))

    # -----------------------
    # local helpers (closures)
    # -----------------------
    def _weights(curve):
        y = np.asarray(curve, float)
        T = y.size
        w = 0.01 + np.exp(-np.clip(y, -1e6, 1e6))
        ttp = int(np.argmax(y))
        if 0 <= ttp < T:   w[ttp]      = max(w[ttp] / 10.0, 1e-6)
        if 0 <= ttp-1 < T: w[ttp - 1]  = max(w[ttp - 1] / 5.0,  1e-6)
        if 0 <= ttp+1 < T: w[ttp + 1]  = max(w[ttp + 1] / 2.0,  1e-6)
        return w

    def _gv(p, t):
        t0, alpha, beta, A = [float(v) for v in p]
        t = np.asarray(t, float)
        y = np.zeros_like(t, float)
        m = t > t0
        x = np.maximum(t[m] - t0, 1e-12)
        y[m] = A * (x ** alpha) * np.exp(-x / (beta + 1e-12))
        return y

    def _fit_gv_peak1(y, w, t):
        y = np.asarray(y, float); w = np.asarray(w, float); t = np.asarray(t, float)
        T = y.size; MC = float(np.max(y)); ttp_idx = int(np.argmax(y)); TTP = float(t[ttp_idx])

        pre = np.where(y[:ttp_idx] <= 0.05 * MC)[0]
        if pre.size:
            t0_init = float(t[int(pre[-1])])
        else:
            base_end = max(1, T // 3)
            base_mean = float(np.mean(y[:base_end])); base_std  = float(np.std(y[:base_end]) + 1e-6)
            pre2 = np.where(y[:ttp_idx] <= base_mean + 2.0 * base_std)[0]
            t0_init = float(t[int(pre2[-1])]) if pre2.size else float(t[0])

        # end of first-pass fit region
        i = ttp_idx
        while i < T and y[i] > (0.2 * MC): i += 1
        end = min(max(ttp_idx + 1, i), T)

        t_fit = t[:end]; y_fit = y[:end]; w_fit = w[:end] if w.shape == y.shape else np.ones_like(y_fit)

        y_tmp = _gv([t0_init, 5.0, max((TTP - t0_init) / (5.0 + 1e-6), 0.25), 1.0], t_fit)
        y_tmp_max = float(np.max(y_tmp)) if np.any(y_tmp > 0) else 1e-6
        A_init = max(MC / y_tmp_max, 1e-6)
        p0 = np.array([t0_init, 5.0, max((TTP - t0_init) / (5.0 + 1e-6), 0.25), A_init], float)
        lb = np.array([t[0] - (t[1]-t[0] if t.size > 1 else 0.5), 0.5, 0.1, 0.0], float)
        ub = np.array([TTP, 20.0, 20.0, np.inf], float)

        def resid(p): return (_gv(p, t_fit) - y_fit) / (w_fit + 1e-6)

        res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=1000, xtol=1e-8, gtol=1e-8, ftol=1e-8)
        return np.asarray(res.x, float)

    def _fit_bat_and_A(y, w, alpha, beta, t0_hint, A_hint, t):
        # solve for [t0, A] given fixed (alpha, beta)
        y = np.asarray(y, float); w = np.asarray(w, float); t = np.asarray(t, float)
        T = y.size; MC = float(np.max(y)); ttp_idx = int(np.argmax(y))
        i = ttp_idx
        while i < T and y[i] > (0.2 * MC): i += 1
        end = min(max(ttp_idx + 1, i), T)
        t_fit = t[:end]; y_fit = y[:end]; w_fit = w[:end] if w.shape == y.shape else np.ones_like(y_fit)

        # init t0
        pre = np.where(y[:ttp_idx] <= 0.05 * MC)[0]
        if pre.size: t0_init = float(t[int(pre[-1])])
        else:
            base_end = max(1, T // 3)
            base_mean = float(np.mean(y[:base_end])); base_std = float(np.std(y[:base_end]) + 1e-6)
            pre2 = np.where(y[:ttp_idx] <= base_mean + 2.0 * base_std)[0]
            t0_init = float(t[int(pre2[-1])]) if pre2.size else float(t[0])
        if t0_hint is not None and np.isfinite(t0_hint): t0_init = float(t0_hint)

        # init A
        if A_hint is not None and np.isfinite(A_hint):
            A_init = float(A_hint)
        else:
            tmp = _gv([t0_init, alpha, beta, 1.0], t_fit)
            tmp_max = float(np.max(tmp)) if np.any(tmp > 0) else 1e-6
            A_init = max(float(np.max(y_fit)) / tmp_max, 1e-6)

        p0 = np.array([t0_init, A_init], float)
        lb = np.array([t[0] - (t[1]-t[0] if t.size > 1 else 0.5), 0.0], float)
        ub = np.array([t[ttp_idx], np.inf], float)

        def resid(p): return (_gv([p[0], alpha, beta, p[1]], t_fit) - y_fit) / (w_fit + 1e-6)

        res = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-8, gtol=1e-8, ftol=1e-8, max_nfev=1000)
        return np.asarray(res.x, float)  # t0, A

    def _fit_alpha_beta(y, w, t0, A, alpha_hint, beta_hint, t):
        # solve for [alpha, beta] given fixed (t0, A)
        y = np.asarray(y, float); w = np.asarray(w, float); t = np.asarray(t, float)
        T = y.size; ttp_idx = int(np.argmax(y)); MC = float(np.max(y))
        i = ttp_idx
        while i < T and y[i] > (0.2 * MC): i += 1
        end = min(max(ttp_idx + 1, i), T)
        t_fit = t[:end]; y_fit = y[:end]; w_fit = w[:end] if w.shape == y.shape else np.ones_like(y_fit)

        alpha_init = float(alpha_hint) if alpha_hint else 5.0
        if beta_hint:
            beta_init = float(beta_hint)
        else:
            pre = np.where(y[:ttp_idx] <= 0.05 * MC)[0]
            if pre.size:
                t0_est = float(t[int(pre[-1])])
            else:
                base_end = max(1, T // 3)
                base_mean = float(np.mean(y[:base_end])); base_std  = float(np.std(y[:base_end]) + 1e-6)
                pre2 = np.where(y[:ttp_idx] <= base_mean + 2.0 * base_std)[0]
                t0_est = float(t[int(pre2[-1])]) if pre2.size else float(t[0])
            beta_init = max((float(t[ttp_idx]) - t0_est) / (alpha_init + 1e-6), 0.25)

        p0 = np.array([alpha_init, beta_init], float)
        lb = np.array([0.5, 0.1], float); ub = np.array([20.0, 20.0], float)

        def resid(p): return (_gv([t0, p[0], p[1], A], t_fit) - y_fit) / (w_fit + 1e-6)

        res = least_squares(resid, p0, bounds=(lb, ub), xtol=1e-8, gtol=1e-8, ftol=1e-8, max_nfev=1000)
        return np.asarray(res.x, float)  # alpha, beta

    # -----------------------
    # ROI and per-voxel fits
    # -----------------------
    rows_nonzero = np.where(mask.sum(axis=1) > 0)[0]
    cols_nonzero = np.where(mask.sum(axis=0) > 0)[0]
    if rows_nonzero.size == 0 or cols_nonzero.size == 0:
        raise ValueError("Mask slice is empty; cannot compute AIF.")

    ROI = (mask & vesselSlice) if (vesselSlice is not None and vesselSlice.any()) else mask
    if ROI.sum() == 0:
        raise ValueError("ROI is empty (mask∩vessels is zero).")

    rr, cc = np.where(ROI)
    fit_Params = np.zeros((nR, nC, 4), float)  # [t0, alpha, beta, A]

    for r, c in zip(rr, cc):
        y = AIFslice[r, c, :].astype(float)
        p = _fit_gv_peak1(y, _weights(y), time)
        fit_Params[r, c, :] = p

    # Iteratively refine by alternating (t0,A) and (alpha,beta), update population means
    tmp = np.zeros_like(fit_Params)

    def _avg_ab(M, maskAIF):
        idx = np.where(maskAIF)
        return float(np.mean(M[idx[0], idx[1], 1])), float(np.mean(M[idx[0], idx[1], 2]))

    avg_alpha, avg_beta = _avg_ab(fit_Params, ROI)
    for _ in range(20):
        for r, c in zip(rr, cc):
            y = AIFslice[r, c, :].astype(float)
            w = _weights(y)
            t0, A = _fit_bat_and_A(y, w, avg_alpha, avg_beta, tmp[r, c, 0], tmp[r, c, 3], time)
            alpha, beta = _fit_alpha_beta(y, w, t0, A, avg_alpha, avg_beta, time)
            tmp[r, c, :] = [t0, alpha, beta, A]
        avg_alpha, avg_beta = _avg_ab(tmp, ROI)

    # Choose voxel with minimal t0 as final AIF
    best = None
    best_t0 = np.inf
    for r, c in zip(rr, cc):
        t0 = tmp[r, c, 0]
        if np.isfinite(t0) and t0 < best_t0:
            best_t0 = t0
            best = (r, c)

    if best is None:
        raise RuntimeError("Failed to identify AIF voxel.")

    r, c = best
    p = tmp[r, c, :].copy()
    aif_fit = {
        "parameters": p,
        "gv": _gv(p, time),
        "time": time,
        "conc": AIFslice[r, c, :].astype(float),
    }

    # Sanity check to prevent plotting shape mismatches
    assert (
        aif_fit["conc"].shape == aif_fit["gv"].shape == aif_fit["time"].shape
    ), f"Shape mismatch: conc {aif_fit['conc'].shape}, gv {aif_fit['gv'].shape}, time {aif_fit['time'].shape}"

    # Optional figure (time and conc are guaranteed same length)
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.figure()
        plt.plot(time, aif_fit["conc"], "bo", markersize=4, label="AIF samples")
        plt.plot(time, aif_fit["gv"], "k-", linewidth=2, label="GV fit")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.title("AIF")
        plt.xlim(time[0], time[-1])
        plt.tight_layout()
        plt.savefig(save_fig, dpi=150)
        plt.close()

    return aif_fit
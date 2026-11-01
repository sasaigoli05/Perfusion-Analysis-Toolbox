#!/usr/bin/env python3
"""
Bayesian (MAP) parametric deconvolution for perfusion.

Why this exists
---------------
Truncated SVD treats the residue function as T free numbers and fights the
resulting ill-conditioning with two blunt instruments: discarding small
singular values, and blurring the concentration curves beforehand. Both cost
contrast, and the blur turned out to be load-bearing rather than optional --
measured on a 16-study cohort, five different filter strategies all traded
noise suppression against lesion contrast at a fixed exchange rate, and the
core volume came out at 0.72x the reference no matter which was used.

This module removes the free-form residue instead. If the distribution of
capillary transit times is modelled as a gamma distribution, the residue
function is its complementary CDF and the whole inverse problem collapses to
four parameters per voxel. Regularisation then comes from physiology -- a prior
saying transit times are seconds rather than minutes -- instead of from
smoothing, so contrast survives.

The model
---------
    R(t)   = 1 - P(alpha, t / beta)        complementary regularised
                                           incomplete gamma
    MTT    = alpha * beta
    CTH    = sqrt(alpha) * beta            transit-time heterogeneity
    C(t)   = CBF * (AIF (*) R)(t) * TR

R(0) = 1 by construction, so the fitted amplitude *is* CBF on the same
arbitrary scale as `DSC_SVD` returns -- the two estimators are directly
comparable without rescaling. Tmax is the fitted delay, which is far better
conditioned than reading argmax off a noisy SVD residue.

Why MAP rather than plain least squares
---------------------------------------
A parametric fit by unpenalised least squares was tested first and was *less*
stable than SVD (mean Dice 0.586 vs 0.633, and one study of seven collapsed
outright) even though it recovered core volume far better (ratio 0.80 vs 0.51).
The reason is that a noisy voxel picks whatever transit-time distribution best
fits its noise, with nothing to restrain it. Every failure mode of that
prototype is a missing prior, which is precisely what this adds:

  * a log-normal prior on MTT and on alpha, keeping both physiological;
  * a half-normal prior on the arrival delay, since large delays are rare;
  * a Markov random field prior tying each voxel's *shape* to its in-plane
    neighbours, while leaving the *amplitude* free at full resolution.

That last one matters most and is the whole trick: shape is what is noisy and
genuinely smooth across tissue, amplitude is what carries the lesion contrast.
Regularising them separately is how this avoids paying the blur's price. A
crude version of it (median-filtering the shape maps between passes) already
moved Dice 0.526 -> 0.580 and core ratio 0.67 -> 0.82 on unsmoothed data.

How it is fitted
----------------
Exact MAP over a continuous 4-parameter space per voxel would mean ~1.4 million
nonlinear optimisations. Instead the three nonlinear parameters are discretised
onto a grid, the amplitude is solved in closed form for every grid point at
once -- CBF = (K.c)/(K.K) -- and the posterior is evaluated by one matrix
product. The spatial prior is then handled by iterated conditional modes
(Besag 1986): hold the neighbours fixed, re-solve every voxel, repeat. Three
sweeps is enough; the parameter maps stop moving.

This is a MAP point estimate, not a full posterior. It buys the stability the
priors provide without the cost of sampling, and it is the natural first
implementation to validate before considering a variational or MCMC treatment.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammainc

from .svd import _prepare_aif, _resample_mask_to_conc


# ---------------------------------------------------------------------------
# Parameter grid
#
# Physiological ranges, log-spaced where the parameter acts multiplicatively.
# MTT below ~1s is shorter than one frame and cannot be resolved; above ~25s is
# not brain tissue. alpha is the shape of the transit-time distribution:
# alpha=1 is a single well-mixed compartment (exponential residue), large alpha
# approaches a single sharp transit time, and small alpha is extreme
# heterogeneity.
# ---------------------------------------------------------------------------
MTT_GRID = np.array(
    [1.0, 1.6, 2.3, 3.2, 4.3, 5.7, 7.5, 10.0, 13.5, 18.0, 25.0, 35.0, 50.0]
)
# The upper end matters more than it looks. alpha -> infinity is plug flow --
# every capillary path taking the same time, a box-car residue. That is not
# physiological, but if the grid cannot reach it the fit lands on a shape that
# decays too early, and the amplitude then compensates by inflating: with the
# grid capped at 16, a synthetic box-car residue was recovered with +174% CBF
# error. Extending the ceiling costs three more columns and removes that
# failure mode.
ALPHA_GRID = np.array(
    [0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
)

# Longest arrival delay searched, in seconds. Collateral filling past an
# occlusion is genuinely slow, so this has to be generous.
DEFAULT_MAX_DELAY_S = 12.0

# --- Prior hyperparameters -------------------------------------------------
# Centres and widths of the parameter priors. These are deliberately weak: the
# point is to exclude the implausible, not to pull every voxel toward grey
# matter. Widths are in units of natural log for MTT and alpha, and seconds
# for the delay.
PRIOR_MTT_S = 4.5        # typical parenchymal MTT
PRIOR_MTT_LOG_SD = 0.9   # ~2.5x either side within one SD
PRIOR_ALPHA = 3.0
PRIOR_ALPHA_LOG_SD = 1.1
PRIOR_DELAY_SD_S = 3.0

# Strength of the Markov random field term relative to the likelihood, and the
# neighbourhood radius used to form the local mean. In-plane only: slices are
# thick and shape genuinely changes across them.
DEFAULT_SPATIAL_WEIGHT = 1.0
SPATIAL_RADIUS = 2

# ICM sweeps. The parameter maps are stable well before this; more sweeps
# mainly cost time.
DEFAULT_ITERATIONS = 3

# Gaussian prior on the amplitude (CBF), expressed as a ridge term.
#
# This one is not optional. Profiling the amplitude out of the likelihood
# leaves an objective that rewards maximising (K.c)^2 / ||K||^2, so a kernel
# with a *small norm* that happens to align with the noise wins the selection
# and its small denominator then inflates the amplitude without limit. On a
# synthetic box-car residue this produced +186% CBF error under realistic
# noise while the same fit was accurate to +9% noiseless -- a pure selection
# artifact, not a modelling one. The parameter priors cannot prevent it: with
# 42 frames the RSS/sigma^2 differences run to ~10^3 while those penalties are
# ~1-10, so the likelihood dominates them a hundredfold.
#
# A Gaussian prior on the amplitude turns the closed-form solution into
#     a = (K.c) / (||K||^2 + lambda)
# which shrinks exactly the kernels whose norm is small. Expressed as a
# fraction of the median squared kernel norm it is scale-free, so it behaves
# the same whatever arbitrary units CBF happens to be in: typical kernels are
# barely touched, degenerate ones are damped hard.
#
# Swept on synthetic data against four residue shapes. 0.0005 removes the
# pathology (box-car CBF error +174% -> +16%) while leaving well-specified
# fits essentially untouched (+8.1% -> +7.7%, -1.1% -> -1.2%). Larger values
# start shrinking good amplitudes: at 0.02 the same fits are biased -18% and
# -26%. Weak by design -- it exists to exclude degenerate kernels, not to
# regularise the flow estimate.
AMPLITUDE_RIDGE = 0.0005

# Voxels per chunk when evaluating the posterior. Bounds peak memory at
# roughly chunk * n_grid * 4 bytes.
CHUNK = 100_000


def residue(t: np.ndarray, mtt: float, alpha: float) -> np.ndarray:
    """
    Residue function for gamma-distributed transit times.

    R(t) = 1 - P(alpha, t/beta) with beta = MTT/alpha, so R(0) = 1 and R decays
    monotonically. `gammainc` in SciPy is already the *regularised* lower
    incomplete gamma, i.e. the gamma CDF, which is exactly what is needed here.
    """
    beta = mtt / alpha
    return 1.0 - gammainc(alpha, t / max(beta, 1e-9))


def _delay_grid(TR: float, max_delay_s: float) -> np.ndarray:
    """Arrival delays in whole frames, up to max_delay_s."""
    n = int(np.floor(float(max_delay_s) / max(TR, 1e-9)))
    return np.arange(0, max(n, 0) + 1)


def build_kernel_bank(aif: np.ndarray, TR: float, nT: int,
                      max_delay_s: float = DEFAULT_MAX_DELAY_S):
    """
    Design matrix of AIF-convolved residue functions.

    Returns (K, params) where K is (nT, n_grid) with unit-free columns and
    params is (n_grid, 3) holding MTT, alpha and delay-in-seconds for each
    column. Building this once and reusing it for every voxel is what makes a
    per-voxel nonlinear fit affordable.
    """
    t = np.arange(nT, dtype=float) * TR
    delays = _delay_grid(TR, max_delay_s)

    cols, params = [], []
    for d in delays:
        aif_d = np.concatenate([np.zeros(int(d)), aif])[:nT] if d else aif
        for mtt in MTT_GRID:
            for alpha in ALPHA_GRID:
                k = np.convolve(aif_d, residue(t, mtt, alpha))[:nT] * TR
                if np.linalg.norm(k) < 1e-12:
                    continue
                cols.append(k)
                params.append((mtt, alpha, d * TR))

    if not cols:
        raise ValueError("kernel bank is empty; check the AIF")

    return (
        np.asarray(cols, dtype=np.float32).T,
        np.asarray(params, dtype=np.float32),
    )


def estimate_noise_variance(conc: np.ndarray, mask: np.ndarray) -> float:
    """
    Per-voxel temporal noise variance, from second differences.

    A concentration curve is smooth in time while its noise is not, so the
    second difference is dominated by noise. For white noise of variance s^2
    the second difference has variance 6 s^2, which inverts to a robust
    estimate that needs no model of the signal. The median across voxels keeps
    it from being set by the few voxels containing vessels.
    """
    if conc.shape[-1] < 4 or not mask.any():
        return 1.0
    d2 = np.diff(conc[mask], n=2, axis=-1)
    var = np.median(np.var(d2, axis=-1)) / 6.0
    return float(max(var, 1e-6))


def _parameter_penalty(params: np.ndarray) -> np.ndarray:
    """
    -2 log p(theta) for the parameter priors, up to an additive constant.

    Log-normal on MTT and alpha because both are scale parameters -- a factor
    of two too large should cost the same as a factor of two too small.
    Half-normal on delay, which is non-negative and usually small.
    """
    mtt, alpha, delay = params[:, 0], params[:, 1], params[:, 2]
    pen = ((np.log(mtt) - np.log(PRIOR_MTT_S)) / PRIOR_MTT_LOG_SD) ** 2
    pen += ((np.log(alpha) - np.log(PRIOR_ALPHA)) / PRIOR_ALPHA_LOG_SD) ** 2
    pen += (delay / PRIOR_DELAY_SD_S) ** 2
    return pen.astype(np.float32)


def _neighbour_mean(field: np.ndarray, mask: np.ndarray,
                    radius: int = SPATIAL_RADIUS) -> np.ndarray:
    """
    Mask-aware in-plane mean of a parameter field.

    Averaging must not pull values in from outside the brain, so both the field
    and an indicator are accumulated and divided -- the standard normalised
    box-filter trick for masked data. Uniform filtering is separable and
    therefore cheap enough to run inside the ICM loop.
    """
    from scipy.ndimage import uniform_filter

    size = (1, 2 * radius + 1, 2 * radius + 1)
    m = mask.astype(np.float32)
    num = uniform_filter(np.where(mask, field, 0.0).astype(np.float32),
                         size=size, mode="nearest")
    den = uniform_filter(m, size=size, mode="nearest")
    out = np.zeros_like(field, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 1e-6)
    # Voxels with no valid neighbours keep their own value rather than zero.
    out[den <= 1e-6] = field[den <= 1e-6]
    return out


def DSC_bayes(
    conc,
    aif_curve,
    mask,
    TR=1.5,
    aif_time=None,
    max_delay_s=DEFAULT_MAX_DELAY_S,
    iterations=DEFAULT_ITERATIONS,
    spatial_weight=DEFAULT_SPATIAL_WEIGHT,
    prior_weight=1.0,
    amplitude_ridge=None,
    verbose=True,
):
    """
    MAP parametric deconvolution.

    Returns (cbf_map, tmax_map, extras) where extras carries the maps this
    estimator produces that SVD structurally cannot: MTT and CTH come directly
    from the fitted transit-time distribution rather than from dividing one
    noisy map by another.
    """
    conc = np.asarray(conc, dtype=np.float32)
    if conc.ndim != 4:
        raise ValueError(f"conc must be (Z,Y,X,T), got {conc.shape}")
    nZ, nY, nX, nT = conc.shape

    mask_rs = _resample_mask_to_conc(mask, (nZ, nY, nX))
    aif = np.asarray(_prepare_aif(aif_curve, aif_time, nT, TR), dtype=np.float32)

    K, params = build_kernel_bank(aif, TR, nT, max_delay_s)
    Kn_raw = (K * K).sum(axis=0)                    # (G,)
    ridge_frac = (
        AMPLITUDE_RIDGE if amplitude_ridge is None else float(amplitude_ridge)
    )
    ridge = ridge_frac * float(np.median(Kn_raw))
    Kn = Kn_raw + ridge                             # amplitude prior
    sigma2 = estimate_noise_variance(conc, mask_rs)
    par_pen = _parameter_penalty(params) * float(prior_weight)

    if verbose:
        print(
            f"[BAYES] grid={K.shape[1]} kernels, ridge={ridge:.4g} "
            f"({len(MTT_GRID)} MTT x {len(ALPHA_GRID)} alpha x "
            f"{len(_delay_grid(TR, max_delay_s))} delay), "
            f"noise var={sigma2:.4g}, {iterations} ICM sweeps"
        )

    idx = np.flatnonzero(mask_rs.ravel())
    if idx.size == 0:
        zeros = np.zeros((nZ, nY, nX), np.float32)
        return zeros, zeros.copy(), {}

    C = conc.reshape(-1, nT)[idx]                   # (N, T)
    c_energy = (C * C).sum(axis=1, keepdims=True)   # (N, 1)

    # Shape parameters are regularised in log space for MTT and alpha, matching
    # the priors, and in seconds for the delay.
    grid_feat = np.stack(
        [np.log(params[:, 0]), np.log(params[:, 1]), params[:, 2]], axis=1
    ).astype(np.float32)                             # (G, 3)
    feat_sd = np.array(
        [PRIOR_MTT_LOG_SD, PRIOR_ALPHA_LOG_SD, PRIOR_DELAY_SD_S], np.float32
    )

    best = None
    for sweep in range(max(1, int(iterations))):
        if best is None or spatial_weight <= 0:
            spatial_target = None
        else:
            # Neighbour means of the current shape estimate, per voxel.
            cols = []
            for j in range(3):
                vol = np.zeros((nZ, nY, nX), np.float32)
                vol.ravel()[idx] = grid_feat[best, j]
                cols.append(_neighbour_mean(vol, mask_rs).ravel()[idx])
            spatial_target = np.stack(cols, axis=1)   # (N, 3)

        new_best = np.empty(idx.size, np.int32)
        for s in range(0, idx.size, CHUNK):
            c = C[s : s + CHUNK]
            num = c @ K                                       # (n, G)
            # Residual sum of squares with the amplitude profiled out.
            rss = c_energy[s : s + CHUNK] - (num * num) / Kn
            obj = rss / sigma2 + par_pen                      # (n, G)

            if spatial_target is not None:
                tgt = spatial_target[s : s + CHUNK]           # (n, 3)
                for j in range(3):
                    d = (grid_feat[:, j][None, :] - tgt[:, j][:, None]) / feat_sd[j]
                    obj += float(spatial_weight) * (d * d)

            new_best[s : s + CHUNK] = np.argmin(obj, axis=1)

        changed = 1.0 if best is None else float(np.mean(new_best != best))
        best = new_best
        if verbose:
            print(f"[BAYES]   sweep {sweep + 1}: {100 * changed:.1f}% of voxels moved")
        if changed < 0.005:
            break

    # Amplitude for the selected kernel, per voxel. Grouping by kernel keeps
    # this to one dot product per distinct kernel rather than per voxel.
    cbf = np.zeros(idx.size, np.float32)
    for g in np.unique(best):
        sel = best == g
        # Same ridged solution the objective was minimised under, so the
        # reported amplitude is the one that was actually selected for.
        cbf[sel] = np.maximum((C[sel] @ K[:, g]) / Kn[g], 0.0)

    sel_params = params[best]

    def _scatter(values):
        out = np.zeros(nZ * nY * nX, np.float32)
        out[idx] = values
        return out.reshape(nZ, nY, nX)

    cbf_map = _scatter(cbf)
    # The parametric residue decays monotonically from its arrival, so its peak
    # sits exactly at the fitted delay. No argmax over a noisy curve needed.
    tmax_map = _scatter(sel_params[:, 2])
    mtt = sel_params[:, 0]
    alpha = sel_params[:, 1]

    extras = {
        "mtt_parametric": _scatter(mtt),
        # Standard deviation of the gamma transit-time distribution:
        # sqrt(alpha)*beta with beta = MTT/alpha, i.e. MTT/sqrt(alpha).
        "cth": _scatter(mtt / np.sqrt(alpha)),
        "alpha": _scatter(alpha),
    }

    if verbose:
        print(
            f"[BAYES] fitted MTT median {np.median(mtt):.2f}s, "
            f"alpha median {np.median(alpha):.2f}, "
            f"delay median {np.median(sel_params[:, 2]):.2f}s"
        )

    return cbf_map, tmax_map, extras

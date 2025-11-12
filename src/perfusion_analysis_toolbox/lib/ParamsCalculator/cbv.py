import numpy as np
from itertools import permutations

def _to_bool_mask(mask):
    """Sanitize to boolean mask (treat NaNs/inf as 0)."""
    m = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    # if not binary, threshold at >0
    if m.dtype != np.bool_:
        m = m > 0
    return m

def _collapse_time_if_needed(mask):
    """If mask is 4D, collapse time by 'any' across the last axis."""
    if mask.ndim == 4:
        # consider any positive label across time as brain
        return (mask > 0).any(axis=-1)
    return mask

def _try_permute_to_shape(arr, target_shape):
    """
    Try all axis permutations of a 3D array to match target_shape (Z,R,C).
    Return permuted array if found, else None.
    """
    if arr.ndim != 3:
        return None
    for p in permutations(range(3)):
        cand = np.transpose(arr, p)
        if cand.shape == target_shape:
            return cand
    return None

def _center_pad_or_crop_z(mask, target_z):
    """
    Pad (with zeros) or center-crop along Z to reach target_z.
    Assumes mask is (Z, R, C) and R, C already match target.
    """
    z, r, c = mask.shape
    if z == target_z:
        return mask

    if z < target_z:
        # pad equal on both ends (front/back); any off-by-one bias to the end
        pad_total = target_z - z
        pad_before = pad_total // 2
        pad_after  = pad_total - pad_before
        return np.pad(mask, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant", constant_values=False)
    else:
        # crop centered
        crop_total = z - target_z
        start = crop_total // 2
        end = start + target_z
        return mask[start:end, :, :]

def _align_mask_to_conc(mask, conc):
    """
    Make mask 3D and aligned to conc's first 3 dims (Z,R,C).
    Strategies:
      - collapse time if mask is 4D
      - try permutations of axes
      - if Z mismatch (but R,C match), center pad/crop Z
      - if Z=1, broadcast along Z
    """
    mask = _collapse_time_if_needed(mask)
    mask = _to_bool_mask(mask)

    # conc expected shape (Z,R,C,T)
    if conc.ndim != 4:
        raise ValueError(f"Expected conc to be 4D, got shape {conc.shape}")

    Z, R, C, _ = conc.shape

    # Easy case: already matches
    if mask.shape == (Z, R, C):
        return mask

    # If mask is 2D and matches (R,C), broadcast across Z
    if mask.ndim == 2 and mask.shape == (R, C):
        return np.broadcast_to(mask, (Z, R, C))

    # If mask is 3D:
    if mask.ndim == 3:
        # 1) Perfect shape after permutation?
        perm = _try_permute_to_shape(mask, (Z, R, C))
        if perm is not None:
            return perm

        # 2) If (R,C) match but Z differs, fix Z
        # Try identifying which axes correspond to R,C:
        # check all permutations and take the one that matches R,C on last two dims
        for p in permutations(range(3)):
            cand = np.transpose(mask, p)
            if cand.shape[1:] == (R, C):
                return _center_pad_or_crop_z(cand, Z)

        # 3) If Z=1 along some axis and (R,C) match after perm, broadcast Z
        for p in permutations(range(3)):
            cand = np.transpose(mask, p)
            if cand.shape[0] == 1 and cand.shape[1:] == (R, C):
                return np.broadcast_to(cand, (Z, R, C))

        # 4) Last resort: if mask has same total voxel count as a single slice (R*C),
        # try reshaping slice-by-slice uniformly (broadcast across Z)
        if mask.size == R * C:
            m2 = mask.reshape(R, C) if mask.shape != (R, C) else mask
            return np.broadcast_to(m2, (Z, R, C))

        # 5) Fallback: center crop/pad all three dims independently (less ideal, but avoids crash).
        # Compute per-dim adjustments.
        def _pad_or_crop_dim(arr, dim, tgt):
            sz = arr.shape[dim]
            if sz == tgt:
                return arr
            if sz < tgt:
                pad_total = tgt - sz
                pre = pad_total // 2
                post = pad_total - pre
                pad_width = [(0,0)] * arr.ndim
                pad_width[dim] = (pre, post)
                return np.pad(arr, pad_width, mode="constant", constant_values=False)
            else:
                drop = sz - tgt
                start = drop // 2
                end = start + tgt
                sl = [slice(None)] * arr.ndim
                sl[dim] = slice(start, end)
                return arr[tuple(sl)]

        m3 = mask.copy()
        # try to get last two dims (R,C) first by brute pad/crop
        for axis, tgt in [(1, R), (2, C)]:
            if m3.shape[axis] != tgt:
                m3 = _pad_or_crop_dim(m3, axis, tgt)
        # now fix Z
        if m3.shape[0] != Z:
            m3 = _pad_or_crop_dim(m3, 0, Z)
        # Ensure bool
        return m3.astype(bool)

    raise ValueError(f"Unsupported mask shape {mask.shape} for conc {conc.shape}")

def cal(mask, conc, aif, config=None, device=None):
    """
    CBV calculator with robust mask alignment.

    Parameters
    ----------
    mask : array-like (Z,R,C) or (R,C) or (any permutation/4D that can be aligned)
    conc : (Z,R,C,Tc) concentration / perfusion time series
    aif  : dict with keys:
           - "gv"   (Ta,) AIF(t)
           - "time" (Ta,) seconds (uniform or not)
    config : optional object with attributes 'kh' and 'rho'

    Returns
    -------
    cbv : (Z,R,C) float
    """
    # Align mask to conc
    mask_aligned = _align_mask_to_conc(mask, conc)
    mask_aligned = _to_bool_mask(mask_aligned)

    Z, R, C, Tc = conc.shape

    # AIF fields
    aif_gv   = np.asarray(aif["gv"],   dtype=float).reshape(-1)
    aif_time = np.asarray(aif.get("time", np.arange(aif_gv.size, dtype=float)), dtype=float).reshape(-1)

    # Derive dt from AIF time (fallback dt=1.0)
    if aif_time.size > 1 and np.all(np.isfinite(np.diff(aif_time))):
        dt = float(np.median(np.diff(aif_time)))
        dt = dt if dt > 0 else 1.0
    else:
        dt = 1.0

    # time vector for conc
    time_conc = np.arange(Tc, dtype=float) * dt

    conc_safe = np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0)
    # integrate along time (axis=-1)
    num = np.trapz(conc_safe, x=time_conc, axis=-1)  # (Z,R,C)

    den = float(np.trapz(aif_gv, x=aif_time))
    if not np.isfinite(den) or abs(den) < 1e-12:
        den = 1e-12  # prevent divide-by-zero

    kh  = getattr(config, "kh",  1.0) if config is not None else 1.0
    rho = getattr(config, "rho", 1.0) if config is not None else 1.0
    scale = float(kh) / float(rho) if rho != 0 else 1.0

    cbv = np.zeros((Z, R, C), dtype=float)
    cbv[mask_aligned] = scale * (num[mask_aligned] / den)
    # outside mask already zero
    return cbv

import numpy as np

def cal(mask, conc, aif, config=None, device=None):
    """
    mask : (Z,R,C) bool/uint8
    conc : (Z,R,C,Tc) concentration / perfusion time series
    aif  : dict with keys:
           - "gv"   (Ta,)
           - "time" (Ta,)  seconds (uniform or not)
    returns: CBV volume (Z,R,C)
    """
    mask = (mask > 0)
    Z, R, C, Tc = conc.shape

    aif_gv   = np.asarray(aif["gv"],   dtype=float).reshape(-1)
    aif_time = np.asarray(aif.get("time", np.arange(aif_gv.size, dtype=float)), dtype=float).reshape(-1)

    if aif_time.size > 1 and np.all(np.isfinite(np.diff(aif_time))):
        dt = float(np.median(np.diff(aif_time)))
        dt = dt if dt > 0 else 1.0
    else:
        dt = 1.0
    time_conc = np.arange(Tc, dtype=float) * dt

    conc_safe = np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0)
    num = np.trapz(conc_safe, x=time_conc, axis=-1)  # (Z,R,C)

    den = float(np.trapz(aif_gv, x=aif_time))
    if not np.isfinite(den) or abs(den) < 1e-12:
        den = 1e-12  # prevent divide-by-zero

    kh  = getattr(config, "kh",  1.0) if config is not None else 1.0
    rho = getattr(config, "rho", 1.0) if config is not None else 1.0
    scale = float(kh) / float(rho) if rho != 0 else 1.0

    cbv = np.zeros((Z, R, C), dtype=float)
    cbv[mask] = scale * (num[mask] / den)
    cbv[~mask] = 0.0

    return cbv

from .svd import DSC_oSVD

def cal(conc, aif, mask, svd_frac=0.20, max_delay_s=16.0):
    cbf, tmax = DSC_oSVD(
        conc, aif["gv"], mask,
        TR=aif.get("TR", 1.5),
        aif_time=aif.get("time", None),
        svd_frac=svd_frac,
        max_delay_s=max_delay_s,
    )
    return cbf, tmax
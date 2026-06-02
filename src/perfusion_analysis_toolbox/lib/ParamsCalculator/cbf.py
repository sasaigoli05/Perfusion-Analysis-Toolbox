from .svd import DSC_oSVD, DSC_oSVD_adaptive

def cal(conc, aif, mask, svd_frac=0.20, max_delay_s=16.0):
    # EXPERIMENT (experiment/motion-adaptive-svd): adaptive OI regularization.
    # To revert: swap DSC_oSVD_adaptive → DSC_oSVD and restore svd_frac arg.
    cbf, tmax = DSC_oSVD_adaptive(
        conc, aif["gv"], mask,
        TR=aif.get("TR", 1.5),
        aif_time=aif.get("time", None),
        max_delay_s=max_delay_s,
    )
    return cbf, tmax
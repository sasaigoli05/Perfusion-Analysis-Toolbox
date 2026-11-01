"""
CT perfusion analysis toolbox.

Two ways in.

From another pipeline, with arrays already in memory::

    from perfusion_analysis_toolbox import PerfusionSettings, compute_perfusion

    maps = compute_perfusion(raw_4d, brain_mask, vessel_mask,
                             PerfusionSettings(tr=1.5, deconv_method="bayes"))
    maps.cbf, maps.tmax          # numpy arrays, nothing written to disk

From the command line, with NIfTI files::

    python -m perfusion_analysis_toolbox.main --image-path ... --save-path ...

Arrays are (Z, Y, X, T) for the 4D series and (Z, Y, X) for masks, matching
ITK's ordering. `PerfusionSettings()` with no arguments reproduces the
configuration validated against a commercial reference on a 79-study cohort.
"""

__version__ = "0.1.0"

from .settings import PerfusionSettings
from .normalise import to_relative, contralateral_ratio, find_midsagittal

__all__ = [
    "__version__",
    "PerfusionSettings",
    "PerfusionMaps",
    "compute_perfusion",
    "to_relative",
    "contralateral_ratio",
    "find_midsagittal",
    "stages",
]

#: Individual pipeline stages, for callers who want one step rather than the
#: whole pipeline -- a different deconvolution on our AIF, or our MTT on their
#: maps. Each takes plain arrays and writes nothing. Reached as
#: ``perfusion_analysis_toolbox.stages.mtt(cbv, cbf, mask)``.
_STAGE_MODULES = {
    "concentration": ("ctc", "DSC_conc"),
    "detect_cta_artifact": ("ctc", "detect_cta_artifact"),
    "arterial_input": ("aif", "DSC_aif"),
    "deconvolve": ("svd", "cal"),
    "deconvolve_osvd": ("svd", "DSC_oSVD"),
    "deconvolve_bayes": ("bayes", "DSC_bayes"),
    "cbv": ("cbv", "cal"),
    "cbf": ("cbf", "cal"),
    "mtt": ("mtt", "cal"),
}


class _Stages:
    """Lazy accessor so importing the package stays cheap."""

    __slots__ = ()

    def __dir__(self):
        return sorted(_STAGE_MODULES)

    def __getattr__(self, name):
        try:
            module, func = _STAGE_MODULES[name]
        except KeyError:
            raise AttributeError(
                f"no stage {name!r}; available: {sorted(_STAGE_MODULES)}"
            ) from None
        import importlib

        mod = importlib.import_module(
            f".lib.ParamsCalculator.{module}", __name__
        )
        return getattr(mod, func)


stages = _Stages()


def __getattr__(name):
    # The calculator pulls in ITK and SimpleITK, which are slow to import and
    # unnecessary for callers that only want the settings or the
    # normalisation helpers. Deferring keeps `import
    # perfusion_analysis_toolbox` cheap.
    if name in ("compute_perfusion", "PerfusionMaps"):
        from .lib.main_calculator import MainCalculator, PerfusionMaps

        if name == "PerfusionMaps":
            return PerfusionMaps

        def compute_perfusion(raw_4d, brain_mask, vessel_mask,
                              settings=None, save_path=None):
            """
            Run the pipeline and return the maps.

            Parameters
            ----------
            raw_4d : (Z, Y, X, T) array of the CT perfusion series.
            brain_mask, vessel_mask : (Z, Y, X) boolean arrays.
            settings : PerfusionSettings, or None for the validated defaults.
            save_path : optional directory; when given, NIfTIs are written
                there as well as returned. Writing needs reference geometry,
                so `settings.image_path` and `settings.mask_path` must point
                at the source images.

            Returns
            -------
            PerfusionMaps with cbf, cbv, mtt, tmax, ctc, aif and any extras
            the chosen deconvolution produced.
            """
            cfg = settings if settings is not None else PerfusionSettings()
            return MainCalculator(
                raw_perf=raw_4d, mask=brain_mask, vessels=vessel_mask,
                config=cfg, save_path=save_path,
            ).run()

        return compute_perfusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

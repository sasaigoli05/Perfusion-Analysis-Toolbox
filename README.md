# Perfusion-Analysis-Toolbox

Compute perfusion parameter maps (CBF, CBV, MTT, Tmax) from a 4D CT perfusion
series, using an arterial input function extracted from the vessel mask.

This fork adds DICOM ingestion, rigid motion correction, acquisition timing
measured from the DICOM headers, selectable deconvolution (truncated SVD, oSVD,
and a Bayesian parametric fit), automated quality control, and a Python API for
use as a library inside another pipeline.

> Research software. Not validated for quantitative or clinical use.

---

## 🧩 Prerequisites
- **Git**
- **Conda** or **Mamba**
- **Python 3.10**

---

## ⚙️ Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sasaigoli05/Perfusion-Analysis-Toolbox.git
cd Perfusion-Analysis-Toolbox
```

### 2️⃣ Create the environment

```bash
mamba env create -f environment.yml
conda activate perfusion-analysis-toolbox
```

This will install:

* Python 3.10 and scientific dependencies (`numpy`, `scipy`, `pandas`, `matplotlib`, etc.)
* Imaging and I/O libraries (`SimpleITK`, `nibabel`, `pydicom`)
* The **`dcm2niix`** executable for DICOM→NIfTI conversion

### 3️⃣ Install the project

```bash
pip install -e .
```

### 4️⃣ Verify installation

```bash
perfusion-compute-cli -h
```

---

## 🧠 Usage

### Command line

From a DICOM directory — converts the series, measures the frame interval from
the acquisition times, builds the brain and vessel masks, computes the maps and
writes a QC report. Exits non-zero if QC fails.

```bash
python -m perfusion_analysis_toolbox.run_study /path/to/ctp_dicom_dir /path/to/output
```

From NIfTI files you already have:

```bash
perfusion-compute-cli \
  --image-path  4d.nii.gz \
  --mask-path   brain_mask.nii.gz \
  --vessel-path vessel_mask.nii.gz \
  --save-path   out/ \
  --tr 1.5
```

`--tr` is the frame interval in seconds. It is read from the DICOM acquisition
times when available and from the image header otherwise; when neither carries
it, the pipeline stops rather than assuming a value, so pass it explicitly when
working from a bare NIfTI.

Main tuning flags — `-h` lists all of them:

| Flag | Default | |
| ---- | ------- | --- |
| `--deconv-method` | `svd` | `svd`, `osvd`, `osvd_oi`, `bayes` |
| `--svd_frac` | `0.20` | truncated-SVD threshold |
| `--denoise-method` | `gaussian` | `gaussian`, `temporal`, `curvature`, `tips`, `none` |
| `--smooth-fwhm` | `4.0` | in-plane smoothing FWHM, mm |
| `--aif-mask` | — | arterial mask constraining AIF selection |
| `--no-motion-correction` | off | skip rigid motion correction |
| `--kh` / `--rho` | `1.0` | CBV scaling |

### Python

```python
from perfusion_analysis_toolbox import PerfusionSettings, compute_perfusion

maps = compute_perfusion(
    raw_4d,       # (Z, Y, X, T) float array
    brain_mask,   # (Z, Y, X) bool
    vessel_mask,  # (Z, Y, X) bool
    PerfusionSettings(tr=1.5),
)

maps.cbf, maps.cbv, maps.mtt, maps.tmax   # numpy arrays
maps.aif                                  # fitted AIF parameters
```

Nothing is written to disk unless `save_path=` is passed. `PerfusionSettings()`
with no arguments is the default configuration; every CLI flag above is a field
on it.

Individual stages, for callers who want one step rather than the whole
pipeline:

```python
from perfusion_analysis_toolbox import stages

conc, _    = stages.concentration(raw_4d, brain_mask)
aif        = stages.arterial_input(brain_mask, conc, vessel_mask, TR=1.5)
cbf, tmax  = stages.deconvolve(conc, aif, brain_mask, method="bayes")
cbv        = stages.cbv(brain_mask, conc, aif)
mtt        = stages.mtt(cbv, cbf, brain_mask)
```

`dir(stages)` lists them. Each takes plain arrays and writes nothing.

Relative CBF, for thresholding:

```python
from perfusion_analysis_toolbox import to_relative

rcbf, valid = to_relative(maps.cbf, brain_mask)   # against the brain median
core = (rcbf < 0.30) & valid
```

---

## 📊 Outputs

Written to the `--save-path` directory:

| File                | Description                                   |
| ------------------- | --------------------------------------------- |
| `cbf.nii.gz`        | Cerebral Blood Flow map                       |
| `cbv.nii.gz`        | Cerebral Blood Volume map                     |
| `mtt.nii.gz`        | Mean Transit Time map                         |
| `tmax.nii.gz`       | Time-to-peak of the residue function (s)      |
| `ctc.nii.gz`        | Concentration–time curve                      |
| `aif_curve.png`     | Gamma-variate AIF fit                         |
| `aif_params.json`   | Fitted AIF parameters and selected voxel      |

`run_study` additionally writes `provenance.json`, recording the settings, the
measured acquisition timing and the QC verdict.

---

## 🧩 Reproducibility

The environment is defined in [`environment.yml`](./environment.yml):

```bash
conda env create -f environment.yml
conda activate perfusion-analysis-toolbox
```

---

## 📚 References

1. Peruzzo, Denis, et al. “Automatic Selection of Arterial Input Function on Dynamic Contrast-Enhanced MR Images.” *Computer Methods and Programs in Biomedicine*, vol. 104, no. 3, 2011. [https://doi.org/10.1016/j.cmpb.2011.02.012](https://doi.org/10.1016/j.cmpb.2011.02.012)
2. [https://github.com/marcocastellaro/dsc-mri-toolbox](https://github.com/marcocastellaro/dsc-mri-toolbox)
3. [https://github.com/KitwareMedical/ITKTubeTK-CTHead-Perfusion/tree/main](https://github.com/KitwareMedical/ITKTubeTK-CTHead-Perfusion/tree/main)
4. [https://github.com/peirong26/Perfusion-Analysis-Toolbox](https://github.com/peirong26/Perfusion-Analysis-Toolbox)
5. [https://github.com/uncbiag/unigradicon](https://github.com/uncbiag/unigradicon)

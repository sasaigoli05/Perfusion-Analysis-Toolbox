# Perfusion-Analysis-Toolbox
Compute perfusion parameters for a 4D perfusion image using an Arterial Input Function (AIF) extracted from vessel centerlines.

This fork adds:
- A **custom uniGradICON** version (`@unicarl` branch) for AIF slice selection.
- **DICOM ingestion** and conversion with `dcm2niix`.
- A reproducible **Conda environment** for replicating results.

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
* **`uniGradICON`** fork from [uncbiag/unigradicon](https://github.com/uncbiag/unigradicon) (`@unicarl` branch)

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

### Recommended: one command, DICOM to quality-controlled maps

```bash
python -m perfusion_analysis_toolbox.run_study /path/to/ctp_dicom_dir /path/to/output
```

This converts the series, **measures the frame interval from the DICOM
acquisition times**, builds the brain and vessel masks, computes the maps, and
writes a QC report. It exits non-zero if QC fails.

Outputs land in `<output>/maps/`, with `<output>/provenance.json` recording the
settings, the measured acquisition timing and the QC verdict.

### Acquisition timing

The frame interval is measured, never assumed. Timing is built per *volume* —
each frame's time is the median across slice positions — because a
toggling-table 4D acquisition samples different slices at different points in
the cycle, and intervals measured at a single slice location can range from
0.2 s to 2.9 s on a study that is really sampling at 1.5 s.

If the timing cannot be determined, the pipeline **refuses to run** rather than
defaulting. Guessing silently corrupts AIF lead-time gating, Tmax, MTT and CBF
while still producing plausible-looking maps. Supply it explicitly when working
from a bare NIfTI:

```bash
python -m perfusion_analysis_toolbox.run_study /dicom /out --tr 1.5
```

Non-uniform acquisitions (dense early, sparse late) are detected and warned
about. The pipeline models a constant interval, so timing-derived outputs are
approximations on those studies.

### Quality control

```bash
python -m perfusion_analysis_toolbox.qc /path/to/output/maps /path/to/brain_mask.nii.gz
```

These are **plausibility** checks — non-arterial AIF, excessive motion,
implausible brain volume, degenerate maps — not accuracy predictions. That
distinction is measured, not assumed: across a 16-study cohort no
reference-free signal predicted map accuracy, and the strongest apparent
predictor turned out to track lesion size rather than map quality.

### Step by step

To drive the stages separately, or to inspect the masks before computing maps:

```bash
python -m perfusion_analysis_toolbox.preprocess /path/to/dicom_dir /path/to/output
```

writes `image_<ts>.nii.gz`, `acquisition_<ts>.json`, `brain_mask_<ts>.nii.gz`
and `vessel_mask_<ts>.nii.gz` into `<output>/preprocess/`. Then:

```bash
perfusion-compute-cli --image-path <4d.nii.gz> --mask-path <brain_mask.nii.gz> --vessel-path <vessel_mask.nii.gz> --save-path <out>
```

Tuning flags: `--svd_frac` (truncated-SVD threshold, default `0.20`),
`--deconv-method` (`svd`, `osvd`, `osvd_oi`), `--smooth-fwhm` (in-plane
smoothing FWHM in mm, default `4.0`), `--denoise-method`, `--tr`, and `--kh` /
`--rho` for CBV scaling.

### Optional: atlas-based AIF slice selection

The registration-based arterial mask (`lib/ParamsCalculator/register.py`) needs
a reference atlas, which is **not** distributed with this repository. Point it
at your own copy and make sure the `unicarl` conda env is available with CUDA:

```bash
export PERFUSION_ATLAS_PATH=/path/to/atlas.nii.gz
```

When set, the warped ICA-level mask is intersected with the intensity-based
vessel mask to constrain AIF voxel search. If the variable is unset, if
registration fails, or if the overlap is too small, the pipeline falls back to
the intensity-based vessel mask alone. This path is **not validated**.


## 📊 Outputs

All results are written to the `--save-path` directory:

| File                | Description                                   |
| ------------------- | --------------------------------------------- |
| `cbf.nii.gz`        | Cerebral Blood Flow map                       |
| `cbv.nii.gz`        | Cerebral Blood Volume map                     |
| `mtt.nii.gz`        | Mean Transit Time map                         |
| `tmax.nii.gz`       | Time-to-peak of the residue function (s)      |
| `ctc.nii.gz`        | Concentration–time curve                      |
| `aif_curve.png`     | Gamma-variate AIF fit                         |
| `aif_params.json`   | Fitted AIF parameters and selected voxel      |

When atlas registration is enabled, `firstvolume.nii.gz` and the cached
`atlas_trans.hdf5` transform are written alongside them.

---

## 🔬 Validation status

**Research prototype — not validated for quantitative or clinical use.**

Benchmarked against Cercare Medical Neurosuite on an internal cohort of 16 CTP
studies. Cercare is a commercial product with its own modelling assumptions,
not ground truth.

### Choosing the reference

Each export carries two things that could be called "the reference", and they
are not interchangeable:

| Series | What it is |
| ------ | ---------- |
| `parametric-cbf-…` | the vendor's `rCBF [CMN]` map |
| `ct-core-lesion-…` | the vendor's trained **AI** core segmentation |

The AI segmentation is a median **0.28×** the volume of the vendor's own rCBF
map cut at rCBF<0.30. Scoring a threshold-based pipeline against it charges
this pipeline for the vendor model's conservatism, and it makes Dice degenerate
on studies where the AI declares no infarct at all. Results below therefore use
the **vendor rCBF map under the identical rCBF<0.30 rule**; the AI numbers are
reported alongside as a secondary figure.

### Results

Best configuration (vendor-preprocessed input, vendor brain mask,
`--deconv svd`, `svd_frac=0.20`):

| Measure | Result |
| ------- | ------ |
| Spatial correlation of rCBF | r = 0.895 ± 0.050 |
| Core Dice vs vendor rCBF map | 0.552 ± 0.170 |
| Core Dice vs vendor AI segmentation | 0.425 |
| Core volume vs reference | 0.65× (IQR 0.51–0.79) |
| Studies below Dice 0.30 | 2 of 16 |

Across all four input/mask configurations (64 runs), mean Dice against the
rCBF map is 0.457 ± 0.205.

### What this means

**Working:** the spatial pattern of cerebral blood flow tracks the reference
closely and consistently. No study in this cohort fails outright — the
previously reported "1 in 6 fails completely, and which one cannot be
determined in advance" does not reproduce here, and was substantially an
artifact of scoring against the AI segmentation.

**Not working:** core volume is systematically under-called at 0.65× the
reference under an identical threshold rule. The optimal threshold sits at
0.413 rather than the clinical 0.300, and rethresholding recovers +0.123 Dice.

**Do not use Pearson r as a quality gate.** It reads 0.90+ on runs whose Dice
is 0.001. It measures pattern agreement and is insensitive to the scaling
errors that determine whether a threshold lands in the right place.

### Where the residual error comes from

The rCBF distribution is compressed toward its median from both ends —
low percentiles sit ~18% too high, high percentiles ~29% too low. Two
candidate causes have been ruled out:

- **Deconvolution regularisation.** Core volume ratio is invariant across
  `svd_frac` 0.002–0.50 and `osvd_oi` (0.53 / 0.55 / 0.54 / 0.39). The
  shipped default of 0.20 outperforms every alternative tested.
- **Arterial calibration / partial-volume correction.** rCBF normalises by
  the map's own brain median, so any global scale factor — AIF amplitude
  included — cancels exactly and cannot move this number.

The low tail is instead reproduced by **spatial smoothing**: blurring the
vendor's own map by 4–6 mm FWHM in-plane collapses its distribution onto ours
(p5 0.225 vs our 0.228, p10 0.295 vs 0.303, frac<0.30 0.111 vs 0.107). The
pipeline applies `SMOOTH_FWHM_MM = 4.0` before deconvolution. The high tail is
not fully explained by blur and remains open.

### Known limitations

- Core volume under-call is unresolved. Reducing the pre-deconvolution
  smoothing is the untested lead; it trades against noise, which the
  `svd_frac=0.002` experiment showed is severely destructive.
- The gamma-variate AIF refinement loop has no convergence criterion; it runs
  a fixed iteration count and the shape-parameter clamp can absorb divergence
  rather than surfacing it.
- A fit-sanity gate rejects degenerate AIF fits and warns when none survive.
  It prevents a bad fit passing silently but cannot manufacture a good one.
- Atlas-constrained AIF selection is implemented but **not validated** — on an
  earlier cohort it narrowed the candidate pool enough to degrade results. Off
  by default.

## 🧩 Reproducibility

The environment is defined in [`environment.yml`](./environment.yml) and includes:

```yaml
name: perfusion-analysis-toolbox
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy=1.26.*
  - scipy>=1.10
  - pandas>=1.5
  - matplotlib>=3.7
  - tqdm
  - click
  - networkx
  - simpleitk=2.3.*
  - nibabel>=5.0
  - pydicom>=2.4
  - dcm2niix
  - psutil
  - python-dotenv
  - pip:
      - git+https://github.com/uncbiag/unigradicon.git@unicarl
```

To replicate your exact environment:

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
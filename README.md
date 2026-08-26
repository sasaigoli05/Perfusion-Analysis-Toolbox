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

### Option A: Direct NIfTI inputs

If you already have 4D perfusion and 3D mask/vessel NIfTI files:

```bash
perfusion-compute-cli \
  --image-path  /path/to/perfusion_4D.nii.gz \
  --mask-path   /path/to/brain_mask_3D.nii.gz \
  --vessel-path /path/to/vessel_centerlines_3D.nii.gz \
  --save-path   /path/to/output_dir
```

Tuning flags: `--svd_frac` (truncated-SVD threshold, default `0.20`),
`--kh` and `--rho` (haematocrit and tissue-density factors scaling CBV).

### Option B: DICOM inputs

To convert a DICOM study and generate the brain and vessel masks first:

```bash
python -m perfusion_analysis_toolbox.preprocess /path/to/dicom_dir /path/to/output_dir
```

This writes `perfusion_4d.nii.gz`, `brain_mask_<timestamp>.nii.gz` and
`vessel_mask_<timestamp>.nii.gz` into `<output_dir>/preprocess/`. Inspect the
masks, then pass them to `perfusion-compute-cli` as in Option A.

### Optional: atlas-based AIF slice selection

The registration-based arterial mask (`lib/ParamsCalculator/register.py`) needs a
reference atlas, which is **not** distributed with this repository. Point it at
your own copy and make sure the `unicarl` conda env is available with CUDA:

```bash
export PERFUSION_ATLAS_PATH=/path/to/atlas.nii.gz
```

When set, the warped ICA-level mask is intersected with the intensity-based
vessel mask to constrain AIF voxel search. If the variable is unset, if
registration fails, or if the overlap is too small, the pipeline silently falls
back to the intensity-based vessel mask alone.

---

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

Benchmarked against a commercial reference pipeline on an internal cohort of
6 CTP studies. Numbers are agreement with that reference, which is itself a
commercial product with its own modelling assumptions, not ground truth.

| Measure                                     | Result                  |
| ------------------------------------------- | ----------------------- |
| Spatial correlation of rCBF (5 of 6 studies)| r = 0.59 – 0.77         |
| Spatial correlation (1 of 6 studies)        | r ≈ 0 — outright failure |
| Ischemic core overlap, Dice                 | 0.31 mean (0.06 – 0.58) |
| Core volume as fraction of reference        | 30% mean (13% – 62%)    |

### What this means

**Working:** the spatial *pattern* of cerebral blood flow broadly tracks the
reference in 5 of 6 studies — regions of high and low perfusion appear in
approximately the right places.

**Not working:** absolute scale is uncalibrated. Ischemic core volumes are
systematically underestimated by roughly 3× and the core overlap (Dice 0.31)
is well below the ~0.7 that would indicate two segmentations agree. The gap
between good correlation and poor overlap is the signature of a scale offset:
the shape of the perfusion map is approximately right, so any absolute
threshold applied to it lands in the wrong place.

**Unpredictable:** one study in six fails completely, and which one cannot
currently be determined in advance.

### Appropriate use

Suitable for method development, algorithm experimentation, and qualitative
visual comparison.

Not suitable for lesion volumetry, per-patient quantitative endpoints, or any
clinical decision.

### Known limitations

- The gamma-variate AIF refinement loop has no convergence criterion; it runs
  a fixed iteration count and the shape-parameter clamp can absorb divergence
  rather than surfacing it. This is the leading suspect for the failure mode
  above.
- A fit-sanity gate rejects degenerate AIF fits (collapsed amplitude, or
  parameters resting on a clamp bound) and warns when none survive. It
  prevents a bad fit from passing silently, but cannot manufacture a good one
  when the candidate pool is too narrow.
- Atlas-constrained AIF selection is implemented but **not validated** — on
  the cohort above it narrows the candidate pool enough to degrade results.
  It is off by default.
- Systematic core underestimation is unresolved and likely requires arterial
  calibration / partial-volume correction.

---

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
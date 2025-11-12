````markdown
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
````

### 2️⃣ Create the environment

```bash
mamba env create -f environment.yml
conda activate perfusion-analysis-toolbox
```

This will install:

* Python 3.10 and scientific dependencies (`numpy`, `scipy`, `pandas`, `matplotlib`, etc.)
* Imaging and I/O libraries (`SimpleITK`, `nibabel`, `pydicom`)
* The **`dcm2niix`** executable for DICOM→NIfTI conversion
* `uniGradICON`** fork from [uncbiag/unigradicon](https://github.com/uncbiag/unigradicon) (`@unicarl` branch)

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
---

## 📊 Outputs

All results are written to the `--save-path` directory:

| File                                | Description               |
| ----------------------------------- | ------------------------- |
| `CBF.nii.gz`                        | Cerebral Blood Flow map   |
| `CBV.nii.gz`                        | Cerebral Blood Volume map |
| `MTT.nii.gz`                        | Mean Transit Time map     |
| `CTC.nii.gz`                        | Concentration–time curve  |
| `high-res-aif.png`                  | Gamma variate AIF fit     |
| `perfusion_run_YYYYMMDD_HHMMSS.txt` | Run log                   |

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
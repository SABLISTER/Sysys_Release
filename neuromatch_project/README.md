# CORnet-Z vs AlexNet — biological inductive bias as a regularizer (Neuromatch DL)

A self-contained notebook testing whether a **brain-inspired backbone (CORnet-Z)** lets a small
trainable **attention head** learn a stimulus representation that aligns with the visual hierarchy —
and **resists overfitting** small, noisy fMRI data — better than a standard CNN (**AlexNet**).

## The sharpened question
> Does mimicking ventral-stream *hardware* act as a **natural regularizer for the software** — i.e.
> is CORnet-Z's advantage **largest when data is scarce and shrinking as data grows**?

## Method (RSA is the yardstick throughout)
1. **Baseline, no training:** frozen AlexNet & CORnet-Z → per-layer RDMs → align to each ROI's neural
   RDM → a **layer × ROI RSA grid**, read against a split-half noise ceiling.
2. **Intervention (the DL piece):** a self-attention pooling head on the frozen backbone's last
   spatial map, **fine-tuned** with a differentiable RSA loss to match the top ventral ROI's geometry.
3. **Main comparison:** held-out RSA, the **train→held-out gap**, and a **data-size curve spanning two
   datasets** — a narrowing gap with scale is the evidence for "regularizer, not just a better model."

## Two datasets, one adapter — matched V1–V4
The pipeline is dataset-agnostic: each dataset loads behind a common `VisualDataset` interface exposing
its *own* ordered early→late ROIs. Voxels are **never merged across datasets** (different subjects,
scanners, voxel counts) — we align *within* a dataset and compare each one's top ventral ROI. Both
datasets expose retinotopic **V1–V4**, so the cross-dataset curve compares like-for-like (V4→V4).

| Dataset | Images | ROIs (early → late) | Top ventral ROI |
|---|---|---|---|
| **Kay vim-1** (2008) | 1,750 | `V1, V2, V3, V4` | `V4` |
| **NSD** via **Algonauts 2023** | ~9,000 / subject | `V1, V2, V3, V4` (from `prf-visualrois`) | `V4` |

> **No image augmentation.** BOLD responses are stimulus-locked; augmenting the image breaks the
> (image, response) pairing and corrupts the RSA target. A second *real* dataset is the sound way to
> grow N — and it doubles as the scaling axis the hypothesis needs.

> **NSD ROIs:** the Algonauts `prf-visualrois` masks give `V1v/V1d/V2v/V2d/V3v/V3d/hV4`, which the
> loader collapses to `V1/V2/V3/V4` to match Kay exactly.

## Run it
- Open `cornet_z_vs_alexnet_fmri_rsa.ipynb` in Colab, set **Runtime → GPU**, run top-to-bottom.
- **Kay** downloads automatically (verify the OSF URL / npz keys — the loader prints them).
- **NSD** uses the **Algonauts-packaged** data (the tidy `subjXX/` layout), from the Algonauts 2023
  registration / dev-kit **Google Drive** folder — **not** the `natural-scenes-dataset` S3 bucket
  (that is raw NSD: per-session betas, `.mgz` atlases, a ~37 GB stimuli HDF5 — a much heavier layout
  this loader does not read). Either point `ALGONAUTS_ROOT` (cell **1b**) at an extracted `subjXX`
  folder, or set `ALGONAUTS_DOWNLOAD=True` to fetch + unzip it in Colab via `gdown`. Left unset, the
  notebook runs Kay-only and the scaling figure shows the within-Kay curve.

## Key references
- Kay et al. (2008), *Nature* 452:352 — vim-1 dataset.
- Allen et al. (2022), *Nature Neuroscience* 25:116 — NSD.
- Gifford et al. (2023), arXiv:2301.03198 — Algonauts 2023 Challenge.
- Güçlü & van Gerven (2015), *J. Neurosci.* 35:10005 — CNN depth ↔ ventral-stream hierarchy.
- Kriegeskorte et al. (2008) / Nili et al. (2014) — RSA methodology & noise ceilings.
- Kubilius, Schrimpf et al. (2019), *NeurIPS* — CORnet.

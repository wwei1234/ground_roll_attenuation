# Ground Roll Attenuation

This repository contains research code for seismic ground-roll identification,
sample generation, and attenuation experiments.

## Main Components

- `监督学习面波识别/`: supervised U-Net/CBAM segmentation for ground-roll masks.
- `深层数据面波压制/`: deep-data ground-roll attenuation, self-supervised masking, inference, and data utilities.
- `浅层面波压制/`: shallow-data attenuation experiments and supporting processing scripts.
- `Pix2Pix 样本生成/`: conditional Pix2Pix generation from masks to seismic gathers.
- `DDPM样本生成/`: conditional diffusion/DDIM sample generation.
- `LNF/`: local nonlinear filtering, F-K masks, gradient-flow regularization, and velocity analysis.
- `论文投递/`: LaTeX manuscript sources and journal templates.

Large seismic data files, generated figures, checkpoints, logs, and office
documents are intentionally excluded from version control by `.gitignore`.

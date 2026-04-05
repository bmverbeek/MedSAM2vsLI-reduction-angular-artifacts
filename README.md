# Reducing Angular Artifacts in Low z-Resolution Segmentation MedSAM2-Based Slice Propagation vs. Linear Interpolation
This repository contains the code used for the Bachelor End Project (BEP):
“Reducing Angular Artifacts in Low z-Resolution Segmentation: MedSAM2-Based Slice Propagation vs. Linear Interpolation”

In MR-guided radiotherapy, annotating every slice is time-consuming. A common workaround is linear interpolation (LI) between annotated slices, but this introduces angular (staircase) artifacts and discontinuities.

This project evaluates whether MedSAM2-based slice propagation can improve boundary consistency and produce smoother, more anatomically realistic segmentations compared to LI.

Pythion 3.10.19
The MedSAM2 model is publicly available at: https://github.com/bowang-lab/MedSAM2/
The dataset is publicly available at: https://github.com/kbressem/prostate158

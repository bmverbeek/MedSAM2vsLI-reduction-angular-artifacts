# Reducing Angular Artifacts in Low z-Resolution Segmentation MedSAM2-Based Slice Propagation vs. Linear Interpolation
This repository contains the code used for the Bachelor End Project (BEP):
“Reducing Angular Artifacts in Low z-Resolution Segmentation: MedSAM2-Based Slice Propagation vs. Linear Interpolation”

In MR-guided radiotherapy, annotating every slice is time-consuming. A common workaround is linear interpolation (LI) between annotated slices, but this introduces angular (staircase) artifacts and discontinuities.

This project evaluates whether MedSAM2-based slice propagation can improve boundary consistency and produce smoother, more anatomically realistic segmentations compared to LI.

Pipeline:
Install SAM2 using: https://www.youtube.com/watch?v=MIUxiLjoA1g  
Obtain dataset from: https://github.com/kbressem/prostate158  
Obtain MedSAM2 model from: https://github.com/bowang-lab/MedSAM2/  
Convert NIfTI volumes to 2D slices using  : nii_gz_converter.py  
Generate interpolated slices (LI) using   : linear_interpolator.py  
Apply MedSAM2 slice propagation using     : SAM2_propagtion_V4.py  
Extract sagittal and coronal views using  : Sagittal_coronal_constructorV6.py  
Evaluate methods against eachother using  : evaluationV2.py  
Generate overlay visualizations using     : overlayer_sagittal_coronalV5.py  


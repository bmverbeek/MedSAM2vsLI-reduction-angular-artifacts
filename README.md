# Reducing Angular Artifacts in Low z-Resolution Segmentation MedSAM2-Based Slice Propagation vs. Linear Interpolation
This repository contains the code used for the Bachelor End Project (BEP):
“Reducing Angular Artifacts in Low z-Resolution Segmentation: MedSAM2-Based Slice Propagation vs. Linear Interpolation”

## Overview
In MR-guided radiotherapy, annotating every slice is time-consuming. A common workaround is linear interpolation (LI) between annotated slices, but this often introduces angular (staircase) artifacts and discontinuities.  
  
This project evaluates whether MedSAM2-based slice propagation improves boundary consistency and produces smoother, more anatomically realistic segmentations compared to LI.  

Besides visual comparison, three metrics were used:  
- Dice score
- 95th percentile Hausdorff Distance (HD95)
- Average Symmetric Surface Distance (ASSD)

##Pipeline:  
Install SAM2: https://www.youtube.com/watch?v=MIUxiLjoA1g  
Dataset (Prostate158): https://github.com/kbressem/prostate158  
MedSAM2 model: https://github.com/bowang-lab/MedSAM2/  
  
Convert NIfTI volumes to 2D slices using:  
→ nii_gz_converter.py  
  
Generate interpolated slices (LI) using:  
→ linear_interpolator.py  
  
Apply MedSAM2 slice propagation using:  
→ SAM2_propagation_V4.py  
  
Extract sagittal and coronal views using:  
→ Sagittal_Coronal_ConstructorV6.py 
  
Evaluate both methods using:  
→ evaluationV2.py   
  
Generate overlay visualizations using:  
→ Overlayer_sagittal_coronalV5.py  

## Dataset Structure
After running nii_gz_converter.py, the dataset should be structured as follows:
```
path/to/dataset/directory/
    test/
        001/
        002/
        .../
        019/
    train/
        020/
        .../
        158
```
Each folder contains a set of axial images, masks and original t2 nifti files


## Output
Results are saved in the specified directories, including:
- excel file
- overlayed images

## Author
Bob Verbeek

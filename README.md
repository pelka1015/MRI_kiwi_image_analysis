#  MRI Kiwi Image Analysis

A Python-based project for analyzing **3D MRI scans of kiwi fruit**, focusing on detecting and segmenting internal structures such as the **core** and **seeds**.  
The project applies advanced **image processing** techniques including contrast enhancement, morphological filtering, line fitting, radial gradient masking, and region growing to isolate and compare fruit components.

---

## 📋 Features

- Detects and fits the **core axis** of the kiwi  
- Extracts and segments the **core** and **seeds**  
- Compares the **seed mask** against a reference template  
- Uses **NIfTI** image format for all 3D image data  
- Produces multiple segmentation masks for visualization and analysis  

---


##  Method Overview

1. **Preprocessing** — Contrast enhancement and noise reduction  
2. **Morphological Operations** — Clean binary masks of fruit components  
3. **Axis Detection** — Fit a central line through the kiwi core  
4. **Seed Segmentation** — Region-growing and radial gradient masking  
5. **Mask Comparison** — Evaluate segmentation accuracy vs. reference  

Precise explanation:
[presentation.pdf](https://github.com/user-attachments/files/23152903/presentation.pdf)

---

##  Setup

Before running the script, organize your files as follows:
```bash
kiwi_MRI_analysis/
│
├── files/                  
│   ├── kiwi_bergen.nii.gz
│   └── kiwi_bergen_wzorzec.nii
│
├── generated_files/          
│
├── functions_kiwi.py          
├── main.py                    
├── README.md                  
└── __pycache__/               
```
 Make sure you've installed this dependencies

- Python ≥ 3.8
- NumPy 
- SciPy 
- scikit-image 
- scikit-learn 
- OpenCV 
- nibabel 



 Each generated file is a NIfTI (.nii.gz) so it can be viewed by using standard medical image viewers such as:

- ITK-SNAP (I’ve worked with this software)
- 3D Slicer
- MRIcroGL



---

##  Example Visualization

<img width="465" height="499" alt="obraz" src="https://github.com/user-attachments/assets/b9780e63-1658-4e10-87bb-3f4c0e97ef80" />
<img width="465" height="499" alt="obraz" src="https://github.com/user-attachments/assets/f3afbf9d-4845-4052-b284-b2d7b0e68ea8" />
<img width="465" height="499" alt="obraz" src="https://github.com/user-attachments/assets/1804134d-24db-48c0-b081-7a3eb97bc33d" />
<img width="465" height="499" alt="obraz" src="https://github.com/user-attachments/assets/b3b1edb5-8103-4f3e-a2d9-df2ed06babc4" />


---

##  Authors
- Patryk Pełka
- Robert Wewersowicz

---
 Citation

If you use this project in academic work, please cite or reference this repository.

# KC-DIP: Unsupervised Single-Image Super-Resolution for Infant Brain MRI

This repository contains the official implementation of the paper ["Unsupervised single-image super-resolution for infant brain MRI"](https://doi.org/10.1016/j.neuroimage.2025.121293) published in *NeuroImage* (2025).

## Overview
KC-DIP (k-space consistent Deep Image Prior) is an unsupervised single-image super-resolution (SR) framework designed specifically for infant brain MRI. Bypassing the need for large datasets of paired high-resolution (HR) images, KC-DIP requires only a single low-resolution (LR) input for training. By integrating the image-space regularity of DIP with k-space consistency, this approach significantly mitigates overfitting and stabilizes the training process, addressing the critical challenges of long scan times and low subject compliance in pediatric neuroimaging.

## Key Features
* **K-space Consistency**: Employs an exponential weighting strategy during k-space loss computation to balance high- and low-frequency components, preventing model collapse.
* **K-space Replacement (KR) & Boundary Loss ($\mathcal{L}_{KB}$)**: Replaces the generated low-frequency k-space with the original LR data to ensure reconstruction trustworthiness. A hinge-like boundary loss ($\mathcal{L}_{KB}$) guarantees a smooth transition between frequency regions, eliminating discontinuity artifacts.
* **Joint Self-Supervised Learning**: Utilizes a shared-weight 3D U-Net backbone to simultaneously perform unsupervised and self-supervised learning, enhancing the fidelity of low-frequency contextual information.
* **Automated & Stable Optimization**: Achieves fast convergence and maintains high stability post-convergence, completely eliminating the need for manual hyperparameter tuning or early stopping.

## Performance
Evaluated on infant (1 week to 1 year) and adult (HCP) T1w and T2w MRI datasets, KC-DIP significantly outperforms Sinc interpolation and existing single-image SR models (including 3D-DIP and its TV-regularized variant) across PSNR, SSIM, and gradient magnitude similarity deviation (GMSD) metrics.

## Citation
If you find this code or our paper useful for your research, please cite:
```bibtex
@article{tsai2025unsupervised,
  title={Unsupervised single-image super-resolution for infant brain MRI},
  author={Tsai, Cheng Che and Chen, Xiaoyang and Ahmad, Sahar and Yap, Pew-Thian},
  journal={NeuroImage},
  volume={317},
  pages={121293},
  year={2025},
  publisher={Elsevier}
}

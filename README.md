[![codecov](https://codecov.io/gh/TeunHuijben/ALPaCA/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/TeunHuijben/ALPaCA)


# ALPaCA
This repository contains code to calculate the analytical PSF for an emitter near a spherical nanoparticle, as published in: **Point-spread function deformations unlock 3D localization microscopy on spherical nanoparticles [(arXiv)](https://arxiv.org/abs/2306.08148)**. 

The Analytical PSF Calculator (ALPaCA), uses an anaytical model to compute the light intensity on every camera pixel. The image below shows a schematic overview of the process of calculating the PSF of a dipole emitter next to a spherical nanoparticle.
<img src="images/overview_PSF_model.jpg" />
The process is composed of 6 steps: **a** calculation of the EM-fields in the water medium for a standard dipole, fixed in orientation and positioned on top of an NP, in terms of spherical waves, **b** rotation and interpolation of the fields to obtain the desired dipole position/orientation, **c** decomposition of the fields into plane waves, **d** refraction of the plane waves in the water-glass interface, **e** projection of the fields into the far-field, and **f** focusing the fields onto the camera. Schematics are not to scale.

By fitting this PSF model to experimental images of single fluorophores located in 

# Installation
If you want to run ALPaCA locally on your computer, use the following code to clone the repository and install the package. We recommend cloning and installing ALPaCA in a clean conda environment. Alternatively, you can leave out the first two lines, or run the code in via Google Colab (see examples below).

```
conda create -n alpaca python
conda activate alpaca
git clone https://github.com/TeunHuijben/ALPaCA.git
cd alpaca
pip install .
```

# Examples
We have included three examples to get everyone started with the code: 
1. calculate a PSF
2. calculate PSFs for a range of settings
3. fit an experimantel spot


# Citation

If our analytical PSF is usefull for your research, please cite the paper:

```
@article{huijben2023exact,
  title={Exact particle-enhanced point-spread function unlocks 3D super-resolution localization microscopy on nanoparticles},
  author={Huijben, Teun APM and Mahajan, Sarojini and Zijlstra, Peter and Marie, Rodolphe and Mortensen, Kim I},
  journal={arXiv preprint arXiv:2306.08148},
  year={2023}
}
```


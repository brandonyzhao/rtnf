# rtnf

Single-View Refractive Index Tomography using Neural Fields. 

## Installation

This code was built and tested on Ubuntu 22.04 and NVIDIA RTX A6000 GPU. 

There are quite a few packages used in this project. Notably [jax](https://github.com/google/jax) requires a bit of care for installation. Otherwise, packages (output from pip freeze) are included in requirements.txt. After installing jax, you should be able to install these by running 

    pip install -r requirements.txt

However, package management with python can be finicky so you may have to fiddle around with the installation if you run into errors. 

To run neural field recovery, you will have to download a series of simulated measurement files for each experiment. These include data such as camera settings, the true refractive field, simulated refracted image measurements, etc. These can be downloaded at [this link](https://caltech.box.com/s/rg5kvbznj6lii52efjjswlda045bz6fr) (~600MB). After downloading, move the archive into the `exp` directory: 

    mkdir exp
    mv /path/to/exp.tar.gz exp/exp.tar.gz
    tar -xvzf exp/exp.tar.gz 

## Running the code

Included in this repository are a couple of tutorial notebooks for generating true refractive and intensity fields, as well as rendering refracted images, which can be found in the `tutorials` directory. 

There are also three scripts included for neural field recovery, as well as the corresponding simulated measurements from our [paper](https://arxiv.org/abs/2309.04437). They can be run as follows: 

    # Gas flow recovery (Fig. 4): 
    python train_nf_gas.py -b gasflow -e test
    # Recovery of Smooth Refractive Fields (Fig. 5): 
    python train_nf.py -b matern_s2 -e test
    python train_nf.py -b matern_s8 -e test
    python train_nf.py -b matern_s9 -e test
    # Sensitivity to Light Source Density (Fig. 6): 
    python train_nf.py -b src_density_50 -e test
    python train_nf.py -b src_density_100 -e test
    python train_nf.py -b src_density_250 -e test
    # Simulated Dark Matter Halo Recovery (Fig. 1): 
    python train_nf_dm.py -b dark_matter -e test 

These scripts include optional checkpointing functions and training visualization plots using tensorboard. 

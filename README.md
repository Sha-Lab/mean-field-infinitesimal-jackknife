# Uncertainty Estimation with Infinitesimal Jackknife, Its Distribution and Mean-Field Approximation
<img src="teaser/pytorch-logo-dark.png" width="10%">

This repository contains a PyTorch implementation of paper ["Uncertainty Estimation with Infinitesimal Jackknife, Its Distribution and Mean-Field Approximation"](https://arxiv.org/abs/2006.07584).

## Code Structure

- `Utils/`
common util functions for hessian computation, uncertainty metrics evaluation, and etc.
- `MNIST/`
experiments on MNIST dataset.
- `ImageNet/`
experiments on ImageNet dataset.

## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.5.0 (or higher) and torchvision](https://pytorch.org)
- [ipdb](https://pypi.org/project/ipdb/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

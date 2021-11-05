# Perturb-and-max-product: Sampling and learning in discrete energy-based models

This repo contains code for reproducing the results in the paper [Sample-Efficient L0-L2 Constrained Structure Learning of Sparse Ising Models](https://arxiv.org/abs/2012.01744) published at the 35th AAAI Conference on Artificial Intelligence (AAAI 2021).


## Getting started

Dependencies can be installed via

```
pip install -r requirements.txt
python setup.py develop
```
By default this installs JAX for CPU. If you would like to use JAX with a GPU and specific CUDA version (highly recommended), follow the official instructions [here](https://github.com/google/jax#pip-installation-gpu-cuda).
  

## Pmap

`pmap` is the main folder. It contains the following files:
- `mmd.py` implements the maximum mean discrepancy metric.
- `small_ising_scoring.py` contains useful functions for small tractable Ising models.
- `ising_modeling.py` contains learning and sampling algorithms for Ising models using max-product and gibbs variants (in JAX).
- `ising_modeling_lp.py` contains similar algorithms using Ecos LP solver.
- `mplp.py` implements the max-product linear programming algorithm for Ising models.
- `rbm_modeling.py` contains learning and sampling algorithms for RBM models using max-product and gibbs variants (in JAX).
- `rbm_modeling_lp.py` contains similar algorithms using Ecos LP solver.
- `conv_or_modeling.py` and `logical_mpmp.py` contain sampling algorithms for the deconvolution experiments in Section 5.6.


## Experiments

The `experiments` folder contains the python scripts used for all the experiments the paper.

The data required for all the experiments has to be generated first via
```
. experiments/generate_data.sh
```
and will be automatically stored in a `data` folder

- Experiments for Section 5.1 are in `exp1_wrongmodel.py`.
- Experiments for Section 5.2 are in `exp2_mplp.py`.
- Experiments for Section 5.3 are in `exp3_zeros_train.py` and `exp3_zeros_test.py`.
- Experiments for Section 5.4 are in `exp4_c2d_lattice_persistent.py`, `exp4_c2d_lattice_non_persistent.py`, `exp_erdos_persistent.py` and`exp_erdos_non_persistent.py`.
- Experiments for Section 5.5 are in `exp5_mnist_train.py`, `exp5_mnist_test.py` and `exp5_rbm_2s.py`.
- Experiments for Section 5.6 are in `exp6_convor.py`.

The results will be automatically stored in a `results` folder

## Figures

The notebook `all_paper_plots.ipynb` displays all the figures of the main paper.
The figures are saved in a `paper` folder.
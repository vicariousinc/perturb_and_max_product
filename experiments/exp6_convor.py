# Experiments for Section 5.6 - 2D blind deconvolution

import os
import sys

import numpy as np
from jax import random
from pmap.conv_or_modeling import min_energy
from scipy.special import logit as invsigmoid


def solve_convor(X, seed, n_steps=1000):
    rng = random.PRNGKey(seed)

    # Feature and location prior
    pe = 1e-100
    pS = 0.01
    pW = 0.25

    (n_images, n_chan, im_height, im_width) = X.shape
    n_feat, feat_height, feat_width = 5, 6, 6  # larger features, one extra

    s_height, s_width = (
        im_height - feat_height + 1,
        im_width - feat_width + 1,
    )

    # Define the unaries
    uS = invsigmoid(pS) * np.ones((n_images, n_feat, s_height, s_width))
    uW = invsigmoid(pW) * np.ones((n_chan, n_feat, feat_height, feat_width))

    # Unaries of observed variable are very large
    uX = (2 * X - 1) * -invsigmoid(pe)

    # Add perturbation
    rng, rng_input = random.split(rng)
    uS_pert = uS + random.logistic(rng_input, shape=uS.shape)
    rng, rng_input = random.split(rng)
    uW_pert = uW + random.logistic(rng_input, shape=uW.shape)

    # Sample from posterior
    S, W, convergence = min_energy(uX, uS_pert, uW_pert, n_steps)
    W.block_until_ready()
    return S, W, convergence


if __name__ == "__main__":
    seed = int(sys.argv[1])  # 40 to 44

    folder = os.path.join(os.path.dirname(__file__), "../results/convor")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    X = np.load(f"{data_folder}/conv_problem.npz")["X"]

    S, W, convergence = solve_convor(X, seed, n_steps=1000)
    np.savez_compressed(
        f"{folder}/convor_solution_seed_{seed}.npz", S=S, W=W, convergence=convergence
    )

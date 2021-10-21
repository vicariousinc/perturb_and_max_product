# Experiments for Section 5.3 - Ising model: MNIST zeros
# Training

import os
import sys
import time

import numpy as np
from jax import random
from pmap.ising_modeling import learn, mean_corr
from pmap.ising_modeling_lp import learn_lp, np_mean_corr


def train_ising(
    learn_iter,
    n_steps,
    sampling_alg,
    seed,
    n_samples=100,
    reset_chain_on_iter=False,
):
    # Load the 0s from MNIST dataset
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(f"{data_folder}/noisy_mnist.npz")

    X_train = a["X_train"]
    y_train = a["y_train"]
    S = (X_train[y_train == 0] == 0).astype(np.float32).reshape(-1, X_train.shape[1] * X_train.shape[2])

    if sampling_alg == "gibbs_reset":
        reset_chain_on_iter = True
        sampling_alg = "gibbs"

    if sampling_alg == "gwg_reset":
        reset_chain_on_iter = True
        sampling_alg = "gwg"

    if sampling_alg != "pmap":
        # Convert to site updates, since that's what gibbs/gwg uses
        n_steps = n_steps * S.shape[1]

    # Learn the model
    if sampling_alg == "pmap_lp":
        muS, covS = np_mean_corr(S)
        W, b, S2 = learn_lp(
            muS,
            covS,
            learn_iter=learn_iter,
            eta=0.01,
            n_samples=n_samples,
            use_adam=True,
            signed=False,
            seed=seed
        )
    else:
        muS, covS = mean_corr(S)
        W, b, S2 = learn(
            muS,
            covS,
            learn_iter=learn_iter,
            eta=0.01,
            n_samples=n_samples,
            n_steps=n_steps,
            sampling_alg=sampling_alg,
            use_adam=True,
            rng=random.PRNGKey(seed),
            signed=False,
            reset_chain_on_iter=reset_chain_on_iter,
        )
    return W, b


if __name__ == "__main__":
    learn_iter = int(sys.argv[1]) # run with 200
    seed = int(sys.argv[2])  # run with 40 to 44
    n_steps = 50

    folder = os.path.join(os.path.dirname(__file__), "../results/zeros_ising")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    for sampling_alg in ["pmap", "gibbs", "gwg", "gibbs_reset", "gwg_reset", "pmap_lp"]:
        print("Sampling algorithm used: ", sampling_alg)
        start = time.time()
        W, b = train_ising(learn_iter, n_steps, sampling_alg, seed)
        train_time = time.time() - start

        np.savez_compressed(
            f"{folder}/zeros_ising_{learn_iter}_{sampling_alg}_nsteps_{n_steps}_seed_{seed}.npz",
            W=W,
            b=b,
            train_time=train_time
        )

# Experiments for Section 5.5 - Restricted Boltzmann machines, learning and sampling
# Training

import os
import sys

import numpy as np
from jax import random
from pmap.rbm_modeling import learn


def train_rbm(learn_iter, sampling_alg, seed, reset_chain_on_iter=False, n_steps=100):
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(f"{data_folder}/mnist_digits_and_labels.npz")
    X = (a["X_train"].reshape(-1, 28 ** 2) > 0.5).astype(np.float32)  # binary

    # Number of hidden variables
    nh = 500

    if sampling_alg == "gibbs1":
        n_steps = 1
        sampling_alg = "gibbs"
    elif sampling_alg == "gibbs_reset":
        reset_chain_on_iter = True
        sampling_alg = "gibbs"

    # Learn the model
    W, bv, bh, convergence, S = learn(
        X,
        nh,
        mb_size=100,
        learn_iter=learn_iter,
        eta=0.01,
        n_steps=n_steps,
        use_adam=False,
        rng=random.PRNGKey(seed),
        momentum=0.0,
        sampling_alg=sampling_alg,
        reset_chain_on_iter=reset_chain_on_iter,
    )
    return W, bv, bh, convergence


if __name__ == "__main__":
    learn_iter = int(sys.argv[1])  # run with 120000 for 200 epochs with mb_size=100
    seed = int(sys.argv[2])  # 40 to 44

    folder = os.path.join(os.path.dirname(__file__), "../results/rbm_mnist")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    for sampling_alg in ["pmap", "gibbs", "gibbs_reset", "gibbs1", "pmap_lp"]:
        print("Training RBM with", sampling_alg)
        W, bv, bh, convergence = train_rbm(learn_iter, sampling_alg, seed)
        np.savez_compressed(
            f"{folder}/rbm_mnist_noadam_{learn_iter}_{sampling_alg}_seed_{seed}.npz",
            W=W,
            bv=bv,
            bh=bh,
            convergence=convergence,
        )

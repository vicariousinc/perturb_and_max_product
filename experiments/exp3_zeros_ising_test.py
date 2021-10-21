# Experiments for Section 5.3
# Inference

import os
import sys
from tqdm import tqdm

import numpy as np
from jax import random
from pmap.ising_modeling import sample
from pmap.ising_modeling_lp import sample_lp
from pmap.mmd import logMMD


def test_ising(learn_iter, sampling_alg, seed, n_steps=50, n_samples=100):
    # Load the 0s from MNIST dataset
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(f"{data_folder}/noisy_mnist.npz")

    X_train = a["X_train"]
    y_train = a["y_train"]
    S = (X_train[y_train == 0] == 0).astype(np.float32).reshape(-1, X_train.shape[1] * X_train.shape[2])

    # Load the trained model
    folder = os.path.join(os.path.dirname(__file__), "../results/zeros_ising")
    a = np.load(f"{folder}/zeros_ising_{learn_iter}_{sampling_alg}_nsteps_50_seed_{seed}.npz")
    W, b = a["W"], a["b"]

    if sampling_alg != "pmap":
        # Convert to site updates, since that's what gibbs/gwg uses
        n_steps = n_steps * S.shape[1]

    # There are 5923 = 29 * 200 +  123 0s on the MNIST training set
    # We generate the same amount of samples
    logmmd = []
    rng = random.PRNGKey(seed)
    S2 = []
    for j in tqdm(range(30)):
        n_samples = 200 if j < 29 else 123
        if sampling_alg == "pmap_lp":
            S2part, _ = sample_lp(
                W,
                b,
                n_samples,
            )
        else:
            rng, rng_input = random.split(rng)
            S2part, _ = sample(
                W,
                b,
                n_samples,
                n_steps,
                sampling_alg[:-6] if sampling_alg[-6:] == "_reset" else sampling_alg,
                rng_input,
            )
        if sampling_alg == "gibbs" or sampling_alg == "gibbs_reset":
            S2part = S2part[0]
        S2.append(S2part)

    S2 = np.vstack(S2)
    logmmd = logMMD(S, S2)
    print("Log MMD for {}: {}".format(sampling_alg, logmmd))
    return S2, logmmd, W, b


if __name__ == "__main__":
    learn_iter = int(sys.argv[1])  # run with 200
    seed = int(sys.argv[2])  # 40 to 44
    folder = os.path.join(os.path.dirname(__file__), "../results/zeros_ising")

    # Loop over the different methods for different number of steps
    for n_steps in [5, 10, 25, 50, 100]:
        print(f"Sweeping n_steps, {n_steps}")
        for sampling_alg in ["pmap", "gibbs", "gwg", "gibbs_reset", "gwg_reset"]:
            S2, logmmd, W, b = test_ising(
                learn_iter,
                sampling_alg,
                seed,
                n_steps=n_steps,
            )
            np.savez_compressed(
                f"{folder}/zeros_ising_samples_{learn_iter}_{sampling_alg}_nsteps_{n_steps}_seed_{seed}.npz",
                logmmd=logmmd,
                W=W,
                b=b,
                S2=S2,
            )

    # Finish with LP which does not have steps
    S2, logmmd, W, b = test_ising(
        learn_iter,
        "pmap_lp",
        seed,
    )
    for n_steps in [5, 10, 25, 50, 100]:
        np.savez_compressed(
            f"{folder}/zeros_ising_samples_{learn_iter}_pmap_lp_nsteps_{n_steps}_seed_{seed}.npz",
            logmmd=logmmd,
            W=W,
            b=b,
            S2=S2,
        )

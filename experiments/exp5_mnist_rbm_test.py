# Experiments for Section 5.5 - Restricted Boltzmann machines, learning and sampling
# Inference

import os
import sys

import numpy as np
from jax import random
from pmap.mmd import logMMD
from pmap.rbm_modeling import sample


def test_rbm(W, bv, bh, n_steps_sweep, sampling_alg, seed, folder):
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(f"{data_folder}/mnist_digits_and_labels.npz")
    S = (a["X_train"].reshape(-1, 28 ** 2) > 0.5).astype(np.float32)  # binary

    nv = S.shape[1]
    rng = random.PRNGKey(seed)
    rng, rng_input = random.split(rng)
    S = random.permutation(rng_input, S)

    # Samples 10000 digits
    if (
        sampling_alg == "gibbs"
        or sampling_alg == "gibbs_reset"
        or sampling_alg == "gibbs1"
        or sampling_alg == "pcd"
    ):
        n_steps_sweep1 = np.hstack((0, n_steps_sweep))
        rng, rng_input = random.split(rng)
        S2 = np.array(random.bernoulli(rng_input, p=0.5, shape=(10000, nv))).astype(
            np.float32
        )
        for j in range(1, len(n_steps_sweep1)):
            for i in range(10):
                rng, rng_input = random.split(rng)
                S2[i * 1000 : (i + 1) * 1000] = sample(
                    W,
                    bv,
                    bh,
                    1000,
                    n_steps_sweep1[j] - n_steps_sweep1[j - 1],
                    "gibbs" if sampling_alg != "pmap" else sampling_alg,
                    rng_input,
                    S2[i * 1000 : (i + 1) * 1000],
                )[0]

            logmmd = logMMD(S[:10000], S2)
            print(
                "Log MMD for {}, n_steps {}: {}".format(
                    sampling_alg, n_steps_sweep1[j], logmmd
                )
            )

            np.savez_compressed(
                f"{folder}/rbm_mnist_adam_samples_{learn_iter}_{sampling_alg}_nsteps_{n_steps_sweep1[j]}_seed_{seed}.npz",
                logmmd=logmmd,
                W=W,
                bh=bh,
                bv=bv,
                S2=S2,
            )


    elif sampling_alg == "pmap":
        rng, rng_input = random.split(rng)
        S2 = np.array(random.bernoulli(rng_input, p=0.5, shape=(10000, nv))).astype(
            np.float32
        )
        rng, rng_input = random.split(rng)
        for n_steps in n_steps_sweep:
            for i in range(10):
                rng, rng_input = random.split(rng)
                S2[i * 1000 : (i + 1) * 1000] = sample(
                    W,
                    bv,
                    bh,
                    1000,
                    n_steps,
                    sampling_alg,
                    rng_input,
                )[0]

            logmmd = logMMD(S[:10000], S2)
            print("Log MMD for pmap, n_steps {}: {}".format(n_steps, logmmd))

            np.savez_compressed(
                f"{folder}/rbm_mnist_adam_samples_{learn_iter}_{sampling_alg}_nsteps_{n_steps}_seed_{seed}.npz",
                logmmd=logmmd,
                W=W,
                bh=bh,
                bv=bv,
                S2=S2,
            )


if __name__ == "__main__":
    learn_iter = int(sys.argv[1])  # run with 120000 for 200 epochs with mb_size=100
    seed = int(sys.argv[2])  # 40 to 44

    folder = os.path.join(os.path.dirname(__file__), "../results/rbm_mnist")

    n_steps_sweep = np.array([5, 10, 25, 50, 100, 200, 500, 1000, 2000])
    for sampling_alg in ["pmap", "gibbs1", "gibbs_reset", "gibbs"]:
        a = np.load(
            f"{folder}/rbm_mnist_noadam_{learn_iter}_{sampling_alg}_seed_{seed}.npz"
        )
        W = a["W"]
        bv = a["bv"]
        bh = a["bh"]
        test_rbm(W, bv, bh, n_steps_sweep, sampling_alg, seed, folder)

# Experiments for Section 5.5 - Restricted Boltzmann machines, learning and sampling
# Training and testing for 2s only and comparing with LP

import os
import sys
import time

import numpy as np
from jax import random
from pmap.rbm_modeling import learn, sample
from pmap.mmd import logMMD
from pmap.rbm_modeling_lp import learn_lp, sample_lp


def train_rbm(learn_iter, sampling_alg, seed, reset_chain_on_iter=False, n_steps=100):
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(f"{data_folder}/mnist_digits_and_labels.npz")
    X = (a["X_train"][a["y_train"] == 2, :, :]> 0.5).astype(np.float32).reshape(-1, 28 ** 2)[:5000] # binary

    # Number of hidden variables
    nh = 250

    if sampling_alg == "gibbs1":
        n_steps = 1
        sampling_alg = "gibbs"
    elif sampling_alg == "gibbs_reset":
        reset_chain_on_iter = True
        sampling_alg = "gibbs"

    # Learn the model
    if sampling_alg == "pmap_lp":
        W, bv, bh, convergence, S = learn_lp(
            X,
            nh,
            mb_size=50,  # adjust for AWS
            learn_iter=learn_iter,
            eta=0.01,
            use_adam=True,  # changed
            seed=seed
        )
    else:
        W, bv, bh, convergence, S = learn(
            X,
            nh,
            mb_size=50,  # adjust for AWS
            learn_iter=learn_iter,
            eta=0.01,
            n_steps=n_steps,
            use_adam=True,  # changed
            rng=random.PRNGKey(seed),
            momentum=0.0,
            sampling_alg=sampling_alg,
            reset_chain_on_iter=reset_chain_on_iter,
        )
    return W, bv, bh, convergence


def test_rbm(W, bv, bh, n_steps_sweep, sampling_alg, seed, folder):
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(f"{data_folder}/mnist_digits_and_labels.npz")
    S = (a["X_train"][a["y_train"] == 2, :, :]> 0.5).astype(np.float32).reshape(-1, 28 ** 2)[:5000]  # binary
    S = S[:1000]

    nv = S.shape[1]
    rng = random.PRNGKey(seed)
    rng, rng_input = random.split(rng)
    S = random.permutation(rng_input, S)

    # Samples 1000 digits
    if (
        sampling_alg == "gibbs"
        or sampling_alg == "gibbs_reset"
        or sampling_alg == "gibbs1"
        or sampling_alg == "pcd"
    ):
        n_steps_sweep1 = np.hstack((0, n_steps_sweep))
        rng, rng_input = random.split(rng)
        S2 = np.array(random.bernoulli(rng_input, p=0.5, shape=(1000, nv))).astype(
            np.float32
        )
        for j in range(1, len(n_steps_sweep1)):
            rng, rng_input = random.split(rng)
            start = time.time()
            S2 = sample(
                W,
                bv,
                bh,
                1000,
                n_steps_sweep1[j] - n_steps_sweep1[j - 1],
                "gibbs" if sampling_alg != "pmap" else sampling_alg,
                rng_input,
                S2,
            )[0]
            sampling_time = time.time() - start
            logmmd = logMMD(S[:1000], S2)
            print(
                "Log MMD for {}, n_steps {}: {}".format(
                    sampling_alg, n_steps_sweep1[j], logmmd
                )
            )

            np.savez_compressed(
                f"{folder}/rbm_2s_adam_samples_{learn_iter}_{sampling_alg}_nsteps_{n_steps_sweep1[j]}_seed_{seed}.npz",
                logmmd=logmmd,
                W=W,
                bh=bh,
                bv=bv,
                S2=S2,
                sampling_time=sampling_time
            )

    elif sampling_alg == "pmap":
        rng, rng_input = random.split(rng)
        S2 = np.array(random.bernoulli(rng_input, p=0.5, shape=(10000, nv))).astype(
            np.float32
        )
        rng, rng_input = random.split(rng)
        for n_steps in n_steps_sweep:
            rng, rng_input = random.split(rng)
            start = time.time()
            S2 = sample(
                W,
                bv,
                bh,
                1000,
                n_steps,
                sampling_alg,
                rng_input,
            )[0]
            sampling_time = time.time() - start

            logmmd = logMMD(S[:1000], S2)
            print("Log MMD for pmap, n_steps {}: {}".format(n_steps, logmmd))

            np.savez_compressed(
                f"{folder}/rbm_2s_adam_samples_{learn_iter}_{sampling_alg}_nsteps_{n_steps}_seed_{seed}.npz",
                logmmd=logmmd,
                W=W,
                bh=bh,
                bv=bv,
                S2=S2,
                sampling_time=sampling_time
            )

    elif sampling_alg == "pmap_lp":
        start = time.time()
        S2 = sample_lp(
            W,
            bv,
            bh,
            1000,
        )[0]
        sampling_time = time.time() - start

        logmmd = logMMD(S[:1000], S2)
        print("Log MMD for pmap_lp: {}".format(logmmd))

        for n_steps in n_steps_sweep:
            np.savez_compressed(
                f"{folder}/rbm_2s_adam_samples_{learn_iter}_{sampling_alg}_nsteps_{n_steps}_seed_{seed}.npz",
                logmmd=logmmd,
                W=W,
                bh=bh,
                bv=bv,
                S2=S2,
                sampling_time=sampling_time
            )


if __name__ == "__main__":
    learn_iter = int(sys.argv[1])  # run with 1000
    seed = int(sys.argv[2])  # 40 to 44

    folder = os.path.join(os.path.dirname(__file__), "../results/rbm_mnist")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    for sampling_alg in ["pmap", "gibbs", "gibbs_reset", "gibbs1", "pmap_lp"]:
        print("Training RBM with", sampling_alg)
        start = time.time()
        W, bv, bh, convergence = train_rbm(learn_iter, sampling_alg, seed)
        train_time = time.time() - start
        np.savez_compressed(
            f"{folder}/rbm_2s_adam_{learn_iter}_{sampling_alg}_seed_{seed}.npz",
            W=W,
            bv=bv,
            bh=bh,
            convergence=convergence,
            train_time=train_time
        )

    n_steps_sweep = np.array([5, 10, 25, 50, 100, 200, 500, 1000, 2000])
    for sampling_alg in ["pmap", "gibbs1", "gibbs_reset", "gibbs", "pmap_lp"]:
        a = np.load(
            f"{folder}/rbm_2s_adam_{learn_iter}_{sampling_alg}_seed_{seed}.npz"
        )
        W = a["W"]
        bv = a["bv"]
        bh = a["bh"]
        test_rbm(W, bv, bh, n_steps_sweep, sampling_alg, seed, folder)

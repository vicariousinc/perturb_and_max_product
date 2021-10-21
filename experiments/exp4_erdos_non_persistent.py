# Experiments for Section 5.4 - Structured models: 2D lattices and Erdös-Renyi graphs
# Erdös-Renyi graph with non-persistent training

import os
import sys

import numpy as np
from jax import random
from pmap.ising_modeling import learn, mean_corr, sample, stob
from pmap.mmd import logMMD


def run(learn_iter, n_steps, sampling_alg, seed, n_samples=20, signed=True):
    # Get moments from data
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(f"{data_folder}/erdos/erdos_renyi_seed_{seed}.npz")
    Wgt, bgt, S = a["W"], a["b"], a["S"]
    S1 = S
    if signed:
        S1 = 2 * S1 - 1
    muS, covS = mean_corr(S1)

    if sampling_alg != "pmap":
        # Convert to epochs, since that is what pmap uses
        n_steps *= S.shape[0]

    # Learn the model
    W, b, S2 = learn(
        muS,
        covS,
        learn_iter=learn_iter,
        eta=0.001,
        n_samples=n_samples,
        n_steps=n_steps,
        sampling_alg=sampling_alg,
        use_adam=True,
        reg=0.01,
        Wgt=Wgt,
        Sgt=S,
        rng=random.PRNGKey(seed),
        signed=signed,
        reset_chain_on_iter=True,
    )
    if signed:
        Wb, bb = stob(W, b)
    else:
        Wb, bb = W, b

    # Sample from model
    logmmd = []
    rng = random.PRNGKey(seed)
    rng, rng_input = random.split(rng)
    S2, _ = sample(Wb, bb, S.shape[0], n_steps, sampling_alg, rng_input)
    if sampling_alg == "gibbs":
        S2 = S2[0]

    logmmd = logMMD(S, S2)
    print("Log MMD for {}: {}".format(sampling_alg, logmmd))

    logrmse = 0.5 * np.log(((Wgt - W) ** 2).mean())
    print("Log RMSE for {}: {}".format(sampling_alg, logrmse))

    return S2, logmmd, logrmse, Wgt, bgt, W, b


if __name__ == "__main__":
    learn_iter = int(sys.argv[1])  # run with 1000
    seed = int(sys.argv[2])  # 40 to 44

    folder = os.path.join(os.path.dirname(__file__), "../results/erdos")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    for n_steps in [5, 10, 25, 50, 100]:
        print(f"Sweeping n_steps, {n_steps}")
        for sampling_alg in ["pmap", "gibbs", "gwg"]:
            S2, logmmd, logrmse, Wgt, bgt, W, b = run(
                learn_iter, n_steps, sampling_alg, seed
            )
            np.savez_compressed(
                f"{folder}/erdos_slow_li_{learn_iter}_{sampling_alg}_nsteps_{n_steps}_seed_{seed}.npz",
                logmmd=logmmd,
                logrmse=logrmse,
                Wgt=Wgt,
                bgt=bgt,
                W=W,
                b=b,
                S2=S2,
            )

# Experiments for Section 5.4 - Structured models: 2D lattices and Erdös-Renyi graphs
# 2D lattice graph with persistent training

import os
import sys

import numpy as np
from jax import random
from pmap.ising_modeling import learn, mean_corr, sample, stob
from pmap.mmd import logMMD


def run(
    learn_iter, side, theta, n_steps, sampling_alg, seed, n_samples=256, signed=True
):
    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    a = np.load(
        f"{data_folder}/c2d_latt/c2d_latt_theta_{theta}_side_{side}_seed_{seed}.npz"
    )
    Wgt, bgt, S = a["W"], a["b"], a["S"]
    S1 = S
    if signed:
        S1 = 2 * S1 - 1
    muS, covS = mean_corr(S1)

    # Empirical adjustement for PMP to take equal or smaller compute time than Gibbs and GWG
    if sampling_alg == "pmap":
        # Convert to epochs, since that's what pmap uses
        n_samples, n_steps = int(np.floor(4.5 * np.sqrt(n_steps))), 10

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
        # Sgt=S, Not enough memory to handle this one
        rng=random.PRNGKey(seed),
        signed=signed,
    )
    if signed:
        Wb, bb = stob(W, b)
    else:
        Wb, bb = W, b

    # Sample from model
    n_steps = 50
    if sampling_alg != "pmap":
        n_steps = n_steps * S.shape[1]

    logmmd = []
    rng = random.PRNGKey(seed)
    S2 = []
    for _ in range(10):
        rng, rng_input = random.split(rng)
        S2part, _ = sample(Wb, bb, 200, n_steps, sampling_alg, rng_input)
        if sampling_alg == "gibbs":
            S2part = S2part[0]
        S2.append(S2part)

    S2 = np.vstack(S2)
    logmmd = logMMD(S, S2)
    print("Log MMD for {}: {}".format(sampling_alg, logmmd))

    logrmse = 0.5 * np.log(((Wgt - W) ** 2).mean())
    print("Log RMSE for {}: {}".format(sampling_alg, logrmse))

    return S2, logmmd, logrmse, Wgt, bgt, W, b


if __name__ == "__main__":
    learn_iter = int(sys.argv[1])  # run with 100000
    seed = int(sys.argv[2])  # 40 to 44

    folder = os.path.join(os.path.dirname(__file__), "../results/c2d_lattice")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    side, theta = 25, -0.1
    for n_steps in [5, 10, 25, 50, 100]:
        print(f"Sweeping n_steps, {n_steps}")
        for sampling_alg in ["pmap", "gibbs", "gwg"]:
            S2, logmmd, logrmse, Wgt, bgt, W, b = run(
                learn_iter, side, theta, n_steps, sampling_alg, seed
            )
            np.savez_compressed(
                f"{folder}/c2dlat_li_{learn_iter}_{sampling_alg}_theta_{theta}_side_{side}_nsteps_{n_steps}_seed_{seed}.npz",
                logmmd=logmmd,
                logrmse=logrmse,
                Wgt=Wgt,
                bgt=bgt,
                W=W,
                b=b,
                S2=S2,
            )

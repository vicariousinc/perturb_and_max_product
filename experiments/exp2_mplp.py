# Experiments for Section 5.2 - MPLP: Max-product is not broken

import os

import numpy as np
from jax import random
from generate_erdos_and_lattice import c2d_latt_mask
from pmap.ising_modeling import min_energy, stob
from pmap.mplp import mplp
from pmap.small_ising_scoring import rand_ising, logZ, logZ_c2dlattice, menergy_pert


def logZboth(W, b, n_samples, rng, n_steps=200):
    # Estimate log-partition function for both Pmap and MPLP
    rng, rng_input = random.split(rng)

    # The difference between two samples from Gumbel follows a logistic
    b_pert = np.array(
        b + random.logistic(rng_input, shape=(n_samples,) + b.shape)
    ).astype(np.float64)

    # Samples for max-product
    Smp = np.heaviside(min_energy(W, b_pert, n_steps, "pmap"), 0.5)[:, :, 0]

    # Samples for MPLP
    Smplp = np.zeros((n_samples, W.shape[1]))
    for i in range(n_samples):
        Smplp[i] = np.heaviside(
            mplp(W, b_pert[i], 50000, damp=1.0, tol=1e-6, sampling_alg="pmap_seq_mplp"), 0.5
        )[:, 0]

    # Upper-bound of the log partition function using samples with Gumbel noise
    # https://arxiv.org/pdf/1206.6410.pdf, Corollary 1
    logZmp = menergy_pert(W, b_pert, Smp)
    logZmplp = menergy_pert(W, b_pert, Smplp)
    return logZmp, logZmplp


def run_ising(sweep_d, n_samples, seed):
    # Pmap vs MPLP on fully connected Ising
    rng = random.PRNGKey(seed)
    logZmp = np.zeros((n_samples, len(sweep_d)))
    logZmplp = np.zeros((n_samples, len(sweep_d)))
    logZexact = np.zeros((n_samples, len(sweep_d)))

    for i, d in enumerate(sweep_d):
        print("Ising model dimension", d)
        for s in range(n_iters):
            rng, rng_input = random.split(rng)
            # Ising model with uniformly sampled weights and biases
            W, b = rand_ising(d=d, c=2.0, f=0.1, rng=rng_input)
            W, b = stob(W, b)

            # Log partition function for Pmap, MPLP
            rng, rng_input = random.split(rng)
            logZmp[s, i], logZmplp[s, i] = logZboth(W, b, 1, rng_input)

            # Exact log partition function
            logZexact[s, i] = logZ(W, b)
    return logZmp, logZmplp, logZexact


def run_c2dlattice(sweep_side, n_samples, theta, seed):
    # Pmap vs MPLP on 2d lattice
    rng = random.PRNGKey(seed)
    logZmp = np.zeros((n_samples, len(sweep_side)))
    logZmplp = np.zeros((n_samples, len(sweep_side)))
    logZexact = np.zeros((n_samples, len(sweep_side)))

    for i, side in enumerate(sweep_side):
        print("2D cyclical lattice side", side)
        mask = c2d_latt_mask(side)
        W, b = mask * theta, np.zeros((side ** 2, 1))
        Wb, bb = stob(W, b)

        # Log partition function for Pmap, MPLP
        rng, rng_input = random.split(rng)
        logZmp[:, i], logZmplp[:, i] = logZboth(Wb, bb, n_samples, rng_input)

        # Exact log partition function for the model in {-1, 1} with E2
        logZ2exact = logZ_c2dlattice(theta, side)
        # Log-partition function for the binary model with E1. The two differ by a constant.
        logZ1exact = logZ2exact + b.sum() - W.sum()
        logZexact[:, i] = logZ1exact
    return logZmp, logZmplp, logZexact


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(__file__), "../results/mplp")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Ising models
    n_iters, seed = 100, 42
    sweep_d = [5, 10, 15]
    logZmp, logZmplp, logZexact = run_ising(sweep_d, n_iters, seed)

    np.savez_compressed(
        folder + "/ising_mplp.npz",
        logZmp=logZmp,
        logZmplp=logZmplp,
        logZexact=logZexact,
    )

    # 2D cyclical square lattice
    theta, side, seed = -0.1, 15, 42
    sweep_side = [5, 10, 15]
    n_samples = 100
    logZmp, logZmplp, logZexact = run_c2dlattice(sweep_side, n_samples, theta, seed)

    np.savez_compressed(
        folder + "/c2dlattice_mplp.npz",
        logZmp=logZmp,
        logZmplp=logZmplp,
        logZexact=logZexact,
    )

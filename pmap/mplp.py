# Max-Product Linear Programming
# https://people.csail.mit.edu/tommi/papers/GloJaa_nips07.pdf

import numba as nb
import numpy as np


@nb.njit
def mplp(W, b, max_iter, damp=1.0, tol=1e-5, sampling_alg="pmap_seq_mplp"):
    np.random.seed(42)
    d = W.shape[0]
    assert b.shape == (d, 1)
    assert W.shape == (d, d)
    assert (np.diag(W) == 0).all()
    assert (W == W.T).all()

    MT = np.zeros((d, d))
    row = np.ones((1, d))
    source, dest = np.triu(np.ones((d, d)), 1).nonzero()
    factors = np.arange(len(source))

    mm = MT @ row.T + b
    for step in range(max_iter):
        delta_max = 0
        np.random.shuffle(factors)

        # Loop over factors
        for f in factors:
            i, j = source[f], dest[f]
            mji_in = mm[j, 0] - MT[j, i]
            # Max-product equation
            mji_out = max(0, mji_in + W[j, i]) - max(0, mji_in)

            mij_in = mm[i, 0] - MT[i, j]
            # Max-product equation
            mij_out = max(0, mij_in + W[i,j]) - max(0, mij_in)

            if sampling_alg == "pmap_seq_mplp":
                mij_out = 0.5 * mij_out - 0.5 * mji_in
                mji_out = 0.5 * mji_out - 0.5 * mij_in

            # Damping
            delta_ij = mij_out - MT[j, i]
            delta_ji = mji_out - MT[i, j]
            delta_max = max(delta_max, abs(delta_ij))
            delta_max = max(delta_max, abs(delta_ji))
            MT[j, i] += damp * delta_ij
            mm[j, 0] += damp * delta_ij
            MT[i, j] += damp * delta_ji
            mm[i, 0] += damp * delta_ji

        mm = MT @ row.T + b  # refresh
        if delta_max < tol:
            break
    else:
        print("BP did not converge in", max_iter, "iterations")
    return mm

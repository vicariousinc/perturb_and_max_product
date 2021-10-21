# Utils for small Ising models

from itertools import product
import numpy as np
from jax import jit
from jax import random
from jax.config import config
from jax.scipy.special import logsumexp
from tqdm import tqdm

config.update("jax_disable_jit", False)

"""
Note: in our experiments we consider two classes of Ising models:
 - models with observations in {0, 1} and energy E1(x) = -0.5 * x.T * W * x - x.T * b,
 as this is traditionally the case
 - models with observations in {-1, 1} and energy E2(x) = - x.T * W * x - x.T * b
 as defined in some work we are comparing our methods with.
"""

#####################################
###### Log partition function #######
#####################################

@jit
def energy(W, b, X):
    # Compute the energy for each sample of an Ising model with observations
    # in {0, 1} and energy E1(x) = -0.5 * x.T * W * x - x.T * b
    # X.shape: N x d
    return -0.5 * ((X @ W) * X).sum(1, keepdims=True) - X @ b


@jit
def menergy_pert(W, b_pert, X):
    # Opposite of the energy of the binary Ising model given unaries perturbations
    return 0.5 * ((X @ W) * X).sum(1) + (X * b_pert[:, :, 0]).sum(1)


def logZ(W, b, max_d=15):
    # Log partition function for small Ising models with obs. in {0, 1} and energy E1
    d = W.shape[0]
    assert d <= max_d, "Dimension too large"
    assert b.shape == (d, 1)
    assert W.shape == (d, d)
    assert (np.diag(W) == 0).all()
    assert (W == W.T).all()

    # All configurations
    Xall = np.array(list(product([0, 1], repeat=d)))

    # Log-partition function
    return logsumexp(-energy(W, b, Xall))


def logZ_c2dlattice(theta, d, max_d=15):
    # Recursive solution to compute the log partition function of a small 2D lattice
    # of size d x d  with obs. in {-1, 1}, energy E2 and w_ij = -theta if i != j, w_ii = 0
    assert d <= max_d, "Dimension too large"

    # All possible configurations for a single column (which is in {-1, 1}^d)
    col_values = np.array(list(product([-1, 1], repeat=d))).astype(float)

    # Shift the configurations
    col_values_shifted = np.empty_like(col_values)
    col_values_shifted[:, 1:] = col_values[:, :-1]
    col_values_shifted[:, 0] = col_values[:, -1]

    # S0 is a diagonal matrix such that S0[i, i] contains the non-normalized probabilities
    # of a sublattice of size d x 1 with configurations col_values[i]
    S0 = np.zeros(shape=(2 ** d, 2 ** d))
    S0[range(2 ** d), range(2 ** d)] = np.exp(2 * theta * (col_values * col_values_shifted).sum(axis=1))

    # T[i, j] contains all the potentials between two configurations col_values[i] and col_values[j]
    T = col_values.dot(col_values.T)
    T = np.exp(2 * theta * T)
    S0_T = S0.dot(T)

    # At iteration t, S[i, j] considers a sublattice of of size d * t and contains the sum of the
    # non-normalized probabilities of all the configurations for which
    # the t_th column is col_values[i] and the first column is col_values[j]
    # We recursively add columns by considering all the new potentials.
    # We do not consider connections between first and last columns, which are added later.
    S = S0
    for _ in tqdm(range(d - 1)):
        S = S0_T.dot(S)

    # Add potentials between first and last columns
    S *= T

    # Return the log partition function
    log_Z1 = np.log(S.sum())
    return log_Z1


#####################################
############# Useful ###############
#####################################

def moments_from_ising(W, b, max_d=14):
    # Moments for a binary Ising model with energy E1
    d = W.shape[0]
    assert d <= max_d, "Dimension too large"
    assert W.shape == (d, d)
    assert b.shape == (d, 1)

    # All configurations
    Xall = np.array(list(product([0, 1], repeat=d)))

    # Probabilities from model
    log_prob = -energy(W, b, Xall) - logsumexp(-energy(W, b, Xall))

    # Moments from model
    mu = (np.exp(log_prob).T @ Xall).T

    # Covariance matrix from model
    C = Xall.T @ (np.exp(log_prob) * Xall)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)
    return mu, C, log_prob


def rand_ising(d, c=1.0, f=1.0, rng=random.PRNGKey(42)):
    # Generate an Ising model with uniform weights in [-c, c] and biases in [-f, f]
    rng, rng_input1, rng_input2 = random.split(rng, 3)
    W, b = (
        2 * c * random.uniform(rng_input1, (d, d)) - c,
        2 * f * random.uniform(rng_input2, (d, 1)) - f,
    )
    W, b = np.array(W), np.array(b)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)  # avoid self-connections
    return W, b


#####################################
########## KL divergences ###########
#####################################

def kl_divergences(log_prob, Wlearned, blearned, S, pc=0.5):
    # KL divergences for Exp 1. This assumes binary data.
    d = S.shape[1]
    # All configurations
    Xall = np.array(list(product([0, 1], repeat=d)))

    # KL divergence between the observed log probabilities and the ones from the model
    kl = (
        np.exp(log_prob)
        * (
            log_prob
            + energy(Wlearned, blearned, Xall)
            + logsumexp(-energy(Wlearned, blearned, Xall))
        )
    ).sum()

    # Log probabilities from the samples
    logq = distr_from_samples(S, pc)

    # KL divergence between the observed probabilities and the ones from the samples
    kl_approx = (np.exp(log_prob) * (log_prob - logq)).sum()
    return kl, kl_approx


def distr_from_samples(S, pc):
    # Estimate the log probabilities from samples
    d = S.shape[1]
    # All configurations
    Xall = np.array(list(product([0, 1], repeat=d)))

    # Hamming distance between samples and valid configurations
    hamming = np.hstack((1 - S, S)) @ np.hstack((Xall, 1 - Xall)).T

    # Get the counts of valid configurations
    pos, c = np.unique(hamming.argmin(1), return_counts=True)
    count = np.zeros(2 ** d, int)
    count[pos] = c

    logp = np.log((count + pc).reshape(-1, 1)) - np.log((count + 0.5).sum())
    return logp



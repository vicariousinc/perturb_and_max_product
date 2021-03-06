# Experiments for Section 5.1 - Learning and sampling from the “wrong” model

import numpy as np
from jax import random
from pmap.ising_modeling import sample, stob, learn_single_param
from pmap.small_ising_scoring import moments_from_ising, kl_divergences

# Our sampling algorithms assume binary data in {0, 1} with E1(x) = - 0.5 * x^T W x - x^T b
# But https://papers.nips.cc/paper/2011/file/cee631121c2ec9232f3a2f028ad5c89b-Paper.pdf
# considers data in {-1, 1} with E2(x) = - theta * \sum_{i<j} x_i x_j.
# We then consider an equivalent reparametrization of E2 via the stob function

theta = 0.5
# Define parameters for data in {-1, 1} and E2
W, b = 0.5 * theta * np.ones((4, 4)), np.zeros((4, 1))
np.fill_diagonal(W, 0)
# Map to data in {0, 1} and E1
W, b = stob(W, b)
# Get moments
muX, covX, log_prob = moments_from_ising(W, b)


# Learn theta with perturb-and-map
theta_pmap = learn_single_param(
    muX,
    covX,
    learn_iter=200,
    eta=0.01,
    n_samples=100,
    n_steps=100,
    rng=random.PRNGKey(42),
)
print("Theta learned with perturb and max-product:", round(theta_pmap, 3))


# Get samples for this value of theta
# Define parameters for data in {-1, 1} and E2
W, b = .5 * theta_pmap * np.ones((4, 4)), np.zeros((4, 1))
np.fill_diagonal(W, 0)
# Map to data in {0, 1} and E1
W, b = stob(W, b)

S = sample(W, b, n_samples=10000, n_steps=100, sampling_alg="pmap")[0]

# Evaluate the samples vs the probabilities from the Gibbs density
klpq_gibbs, klpq_sampling = kl_divergences(log_prob, W, b, S)
print("\nKL(P||Q) (Q from sampling) at theta={} is {}".format(round(theta_pmap, 3), round(klpq_sampling, 4)))
print("KL(P||Q) (Q from sampling) at theta={} is {}".format(round(theta_pmap, 3), round(klpq_gibbs, 4)))

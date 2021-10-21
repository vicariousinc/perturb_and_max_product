# Generate data for Section 5.4 - Structured models: 2D lattices and Erdös-Renyi graphs

import os

import jax.numpy as jnp
import numpy as np
from jax import random
from pmap.ising_modeling import sample, stob
from tqdm import trange


def c2d_latt_mask(side):
    grid = np.arange(side ** 2).reshape(side, side)
    mask = np.zeros((side ** 2, side ** 2))
    mask[grid[:-1], grid[1:]] = 1
    mask[grid[-1], grid[0]] = 1
    mask[grid[:, :-1], grid[:, 1:]] = 1
    mask[grid[:, -1], grid[:, 0]] = 1
    mask = mask + mask.T
    return mask


def sample_dataset(W, b, rng, steps=1000000, n_samples=2000):
    assert steps % 1000 == 0
    S = (
        random.bernoulli(rng, p=0.5, shape=(n_samples, W.shape[0])).astype(jnp.float32),
        0,
    )
    for _ in trange(steps // 1000):
        rng, rng_input = random.split(rng)
        S, _ = sample(W, b, n_samples, 1000, "gibbs", rng_input, S)
    return S[0]


def c2d_lat_generate(theta, side, rng=random.PRNGKey(42)):
    mask = c2d_latt_mask(side)
    W, b = mask * theta, np.zeros((side ** 2, 1))
    Wb, bb = stob(W, b)
    S = sample_dataset(Wb, bb, rng, steps=1000000, n_samples=2000)
    return S, W, b


def er_generate(n=200, n_neighbors=4, rng=random.PRNGKey(42)):
    p = n_neighbors / n
    rng, rng_input = random.split(rng)
    G = np.triu(random.bernoulli(rng_input, p, shape=(n, n)).astype(jnp.float32), 1)
    rng, rng_input = random.split(rng)
    W = 0.5 * G * random.normal(rng_input, shape=(n, n))
    W = W + W.T
    b = 0.1 * np.ones((n, 1))
    Wb, bb = stob(W, b)
    S = sample_dataset(Wb, bb, rng, steps=1000000, n_samples=2000)
    return S, W, b


if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(__file__), "../data")

    for name in ['erdos', 'c2d_latt']:
        this_data_folder = os.path.join(data_folder, name)
        if not os.path.exists(this_data_folder):
            os.makedirs(this_data_folder, exist_ok=True)

    side, theta = 25, -0.1
    seed = 42

    S, W, b = er_generate(n=200, n_neighbors=4, rng=random.PRNGKey(seed))
    np.savez_compressed(
        f"{data_folder}/erdos/erdos_renyi_seed_{seed}.npz", W=W, b=b, S=S
    )
    S, W, b = c2d_lat_generate(theta, side, rng=random.PRNGKey(seed))
    np.savez_compressed(
        f"{data_folder}/c2d_latt/c2d_latt_theta_{theta}_side_{side}_seed_{seed}.npz",
        W=W,
        b=b,
        S=S,
    )
    print("Data generated for 2D lattices and Erdös-Renyi graphs")

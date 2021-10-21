# Generate data for Section 5.6 - 2D blind deconvolution

import os

import numpy as np
from jax import random
from pmap.conv_or_modeling import or_layer


def generate(seed):
    rng = random.PRNGKey(seed)
    n_samples, n_feat, s_height, s_width = 500, 4, 10, 10
    n_chan, feat_height, feat_width = 1, 5, 5

    # Define the 2D binary features
    W = np.zeros((n_chan, n_feat, feat_height, feat_width))
    W[0, 0, 0, :] = W[0, 0, :, 0] = W[0, 0, :, 4] = W[0, 0, 4, :] = 1
    W[0, 1, 2, :] = W[0, 1, :, 2] = 1
    W[0, 2, :5, :5] = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    W[0, 3, :5, :5] = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    # Define the features locations
    # Each feature appear on average once in every image.
    rng, rng_input = random.split(rng)
    S = np.array(
        random.bernoulli(
            rng_input, p=0.01, shape=(n_samples, n_feat, s_height, s_width)
        )
    ).astype(np.float64)

    # Convolve S and W
    X = np.array(or_layer(S, W))

    # Return images with at least two features
    valid = S.sum((1, 2, 3)) > 2
    assert (valid).sum() >= 100
    return X[valid][:100]


if __name__ == "__main__":
    seed = 42

    data_folder = os.path.join(os.path.dirname(__file__), "../data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    X = generate(seed)
    np.savez_compressed(f"{data_folder}/conv_problem.npz", X=X)
    print("Data generated for 2D blind deconvolution experiment")

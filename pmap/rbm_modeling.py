# Learning and sampling for RBM models

import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random
from jax.config import config
from jax.experimental import optimizers
from jax.lax import scan
from jax.nn import sigmoid, softplus
from jax.scipy.special import logsumexp
from tqdm import trange

from .mmd import logMMD

config.update("jax_disable_jit", False)


#####################################
############## Learning #############
#####################################

def learn(
    X,
    nh,
    mb_size=100,
    learn_iter=100,
    sampling_alg="pmap",
    n_steps=100,
    eta=0.1,
    use_adam=True,
    rng=random.PRNGKey(42),
    momentum=0.9,
    reset_chain_on_iter=False,
):
    # Data is binary
    nv = X.shape[1]
    assert mb_size <= X.shape[0]
    assert (X == 0).sum() + (X == 1).sum() == X.size

    # Initialization
    rng, rng_input = random.split(rng)
    W, bv, bh = 0.01 * random.normal(rng_input, shape=(nh, nv)), random.normal(rng_input, shape=(1, nv)), random.normal(rng_input, shape=(1, nh))
    # W, bv, bh = 0.01 * np.random.normal((nh, nv)), np.zeros((1, nv)), np.zeros((1, nh))

    rng, rng_input = random.split(rng)
    Xmb = X[random.permutation(rng_input, X.shape[0]), :].reshape(-1, mb_size, nv)

    if use_adam:
        opt_init, opt_update, get_params = optimizers.adam(eta)
    else:
        opt_init, opt_update, get_params = optimizers.momentum(eta, mass=momentum)
    opt_state = opt_init((W, bv, bh))

    # Learning iterations
    display = {}
    S = None
    convergence = np.zeros(learn_iter)
    pbar = trange(learn_iter)
    for step in pbar:
        rng, rng_input = random.split(rng)
        W, bv, bh = get_params(opt_state)

        if reset_chain_on_iter:
            S = None

        # Get gradients
        gW, gbv, gbh, logZdata, logZmodel, S = grad(
            W, bv, bh, Xmb[step % Xmb.shape[0]], n_steps, sampling_alg, rng_input, S
        )

        # Log likelihood
        log_lik = logZdata.mean() - logZmodel.mean()
        convergence[step] = log_lik
        display["log_lik"] = log_lik
        pbar.set_postfix(display)

        if (step + 1) % (X.shape[0] // mb_size) == 0:
            n_samples = min(1000, X.shape[0])
            S2, _ = sample(W, bv, bh, n_samples, n_steps, sampling_alg)
            logmmd = logMMD(X[:n_samples], S2)
            display["logmmd"] = logmmd
            np.savez_compressed(
                f"../results/rbm_mnist/rbm_mnist_{sampling_alg}_in_progress.npz",
                W=W,
                bv=bv,
                bh=bh,
                convergence=convergence,
            )
        pbar.set_postfix(display)

        # Gradient step
        opt_state = opt_update(step, (-gW, -gbv, -gbh), opt_state)
        W, bv, bh = get_params(opt_state)

    return W, bv, bh, convergence, S


#####################################
############## Sampling #############
#####################################

@jax.partial(jit, static_argnums=(4, 5))  # jit with axis being static
def grad(W, bv, bh, X, n_steps, sampling_alg, rng=random.PRNGKey(42), S=None):
    n_samples = X.shape[0]

    # First term of the gradient
    zdata_W, zdata_bv, zdata_bh = grad_from_samples(W, bh, X)

    # Marginal disitribution of the visible
    # https://christian-igel.github.io/paper/AItRBM-proof.pdf Eq 20
    logitH = X @ W.T + bh
    logZdata = (X * bv).sum(1) + softplus(logitH).sum(1)

    # Sample from model
    S, logZmodel = sample(W, bv, bh, n_samples, n_steps, sampling_alg, rng, S)

    # Second term of the gradient
    zmodel_W, zmodel_bv, zmodel_bh = grad_from_samples(W, bh, S)

    # Gradient
    gW = zdata_W.mean(0) - zmodel_W.mean(0)
    gbv = zdata_bv.mean(0) - zmodel_bv.mean(0)
    gbh = zdata_bh.mean(0) - zmodel_bh.mean(0)

    return gW, gbv, gbh, logZdata, logZmodel, S


@jit
def grad_from_samples(W, bh, X):
    n_samples = X.shape[0]

    # https://christian-igel.github.io/paper/AItRBM-proof.pdf Eq 28, 32, 33
    H = sigmoid(X @ W.T + bh)
    zh, zv = (
        H.reshape(n_samples, 1, -1),
        X.reshape(n_samples, 1, -1),
    )
    Z = zh.transpose((0, 2, 1)) * zv
    return Z, zv, zh


@jax.partial(jit, static_argnums=(3, 4, 5))  # jit with axis being static
def sample(W, bv, bh, n_samples, n_steps, sampling_alg, rng=random.PRNGKey(42), S=None):
    nh = bh.shape[1]
    nv = bv.shape[1]

    bv = jnp.ones((n_samples, 1)) @ bv
    bh = jnp.ones((n_samples, 1)) @ bh

    if sampling_alg == "pmap":
        rng, rng_input = random.split(rng)
        # The difference between two Gumbel follows a logistic
        pert_v = random.logistic(rng_input, shape=(n_samples, nv))
        rng, rng_input = random.split(rng)
        pert_h = random.logistic(rng_input, shape=(n_samples, nh))

        bv_pert = bv + pert_v
        bh_pert = bh + pert_h
        S, logZmodel = min_energy(W, bh_pert, bv_pert, n_steps)

    elif sampling_alg == "gibbs":
        if S is None:
            rng, rng_input = random.split(rng)
            S = random.bernoulli(rng_input, p=0.5, shape=(n_samples, nv)).astype(
                jnp.float32
            )
        rng, rng_input = random.split(rng)
        S, logZmodel = gibbs(W, bh, bv, rng_input, n_steps, S)

    else:
        raise ValueError("Unknown sampling method")
    return S, logZmodel


#####################################
############## Gibbs ################
#####################################


@jax.partial(jit, static_argnums=(4,))  # jit with axis being static
def gibbs(W, bh, bv, rng, n_steps, S):
    nh, nv = W.shape
    n_samples, _ = bh.shape
    assert bh.shape[1] == nh and bv.shape[1] == nv
    assert bv.shape[0] == n_samples

    W = W[None]
    bv = bv[:, None, :]
    bh = bh[:, :, None]

    def update_gibbs(Srng, _):
        S, rng = Srng
        rng, rng_input = random.split(rng)

        # Block updates for Gibbs sampling
        # See https://christian-igel.github.io/paper/AItRBM-proof.pdf, Eq 27
        H = random.bernoulli(
            rng_input,
            p=sigmoid((W * S).sum(2, keepdims=True) + bh),
            shape=(n_samples, nh, 1),
        ).astype(jnp.float32)

        rng, rng_input = random.split(rng)
        S = random.bernoulli(
            rng_input,
            p=sigmoid((W * H).sum(1, keepdims=True) + bv),
            shape=(n_samples, 1, nv),
        ).astype(jnp.float32)
        return (S, rng), None

    S = S[:, None, :]
    S, rng = scan(update_gibbs, (S, rng), None, length=n_steps)[0]

    rng, rng_input = random.split(rng)
    H = random.bernoulli(
        rng_input,
        p=sigmoid(W * S + bh).sum(2, keepdims=True),
        shape=(n_samples, nh, 1),
    )
    S = S[:, 0, :] + 0.0
    H = H[:, :, 0] + 0.0

    # Ogata Tanamura estimator of the log-partition function
    # http://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/NormConstants/PotamianosGoutsiasIEEE1997.pdf
    energy = -(
        ((S @ W[0].T) * H).sum(1) + (S * bv[:, 0, :]).sum(1) + (H * bh[:, :, 0]).sum(1)
    )
    logZ = -logsumexp(energy) + (nh + nv) * jnp.log(2) + jnp.log(n_samples)

    return S, logZ


#####################################
############## Max-product ##########
#####################################

@jax.partial(jit, static_argnums=(3,))  # jit with axis being static
def min_energy(W, bh, bv, n_steps):
    # Max-product for RBM with binary variables
    mp_step = 0.5
    nh, nv = W.shape
    n_samples, _ = bh.shape
    assert bh.shape[1] == nh and bv.shape[1] == nv
    assert bv.shape[0] == n_samples

    W = W[None]
    bv = bv[:, None, :]
    bh = bh[:, :, None]

    # Messages from hidden to visible
    M_to_v = jnp.zeros((n_samples, nh, nv))
    # Messages from visible to hidden
    M_to_h = jnp.zeros((n_samples, nh, nv))
    # Different options for init, use whichever works best

    def update(M, _):
        M_to_v, M_to_h = M
        # Max-product update to hidden
        inc_to_h = M_to_v.sum(1, keepdims=True) + bv - M_to_v
        M_to_h_new = jnp.maximum(0.0, inc_to_h + W) - jnp.maximum(0.0, inc_to_h)

        # Max-product update to visible
        inc_to_v = M_to_h.sum(2, keepdims=True) + bh - M_to_h
        M_to_v_new = jnp.maximum(0.0, inc_to_v + W) - jnp.maximum(0.0, inc_to_v)

        # Damping
        M_to_v = M_to_v + mp_step * (M_to_v_new - M_to_v)
        M_to_h = M_to_h + mp_step * (M_to_h_new - M_to_h)
        return (M_to_v, M_to_h), None

    M_to_v, M_to_h = scan(update, (M_to_v, M_to_h), None, length=n_steps)[0]

    # Compute beliefs
    X = jnp.heaviside(M_to_v.sum(1) + bv[:, 0, :], 0)
    H = jnp.heaviside(M_to_h.sum(2) + bh[:, :, 0], 0)

    # Upper-bound of the log partition function using samples with Gumbel noise
    # https://arxiv.org/pdf/1206.6410.pdf, Corollary 1
    logZ = (
        ((X @ W[0].T) * H).sum(1) + (X * bv[:, 0, :]).sum(1) + (H * bh[:, :, 0]).sum(1)
    )
    return X, logZ


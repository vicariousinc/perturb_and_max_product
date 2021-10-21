# Learning and sampling for Ising models

import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random, vmap
from jax.experimental import optimizers
from jax.lax import dynamic_slice, dynamic_update_slice, scan
from jax.nn import log_softmax, sigmoid
from tqdm import trange

from .mmd import logMMD

# config.update("jax_disable_jit", True)
# config.update("jax_enable_x64", True)


#####################################
############## Learning #############
#####################################

def learn(
    muX,
    covX,
    learn_iter=1000,
    eta=0.1,
    n_samples=100,
    n_steps=50,
    sampling_alg="pmap",
    use_adam=True,
    rng=random.PRNGKey(42),
    momentum=0.9,
    reg=0.0,
    fast=False,
    Wgt=None,
    Sgt=None,
    signed=False,
    learn_bias=True,
    reset_chain_on_iter=False,
    show_every=1000,
):
    # Learn an Ising model in {0, 1} with energy E(x) = - .5 * x^T W x - b^T x
    d = muX.shape[0]
    W, b = jnp.zeros((d, d)), jnp.zeros((d, 1))

    if use_adam:
        opt_init, opt_update, get_params = optimizers.adam(eta)
    else:
        opt_init, opt_update, get_params = optimizers.momentum(eta, mass=momentum)
    opt_state = opt_init((W, b))

    def step(opt_stateSrng, it):
        opt_state, S, rng = opt_stateSrng
        W, b = get_params(opt_state)
        if reset_chain_on_iter:
            rng, rng_input = random.split(rng)
            S = random.bernoulli(rng_input, p=0.5, shape=(n_samples, d)).astype(
                jnp.float32
            )
            if sampling_alg == "gibbs":
                S = (S, 0)
        rng, rng_input = random.split(rng)

        # Get gradients
        gW, gb, S = grad(
            W, b, muX, covX, n_samples, n_steps, sampling_alg, signed, rng_input, S
        )
        if reg > 0.0:
            gW += -reg * jnp.sign(W)
        if not learn_bias:
            gb *= 0

        # Gradient step
        opt_state = opt_update(it, (-gW, -gb), opt_state)
        return (opt_state, S, rng), None

    rng, rng_input = random.split(rng)
    S = random.bernoulli(rng_input, p=0.5, shape=(n_samples, d)).astype(jnp.float32)
    if sampling_alg == "gibbs":
        S = (S, 0)

    display = {}
    logmmd = 0
    logrmse = 0
    if fast:
        opt_state, S, rng = scan(step, (opt_state, S, rng), jnp.arange(learn_iter))[0]
    else:
        # Plot learning
        pbar = trange(learn_iter)
        for it in pbar:
            opt_state, S, rng = step((opt_state, S, rng), it)[0]

            if Wgt is not None:
                W, b = get_params(opt_state)
                logrmse = 0.5 * jnp.log(((Wgt - W) ** 2).mean())
                display["log_rmse"] = f"{logrmse:.{3}f}"
            if Sgt is not None and (it + 1) % show_every == 0:
                W, b = get_params(opt_state)
                if signed:
                    W, b = stob(W, b)
                else:
                    W, b = W, b
                S2, _ = sample(W, b, Sgt.shape[0], n_steps, sampling_alg)
                if sampling_alg == "gibbs":
                    S2 = S2[0]
                logmmd = logMMD(Sgt, S2)
                display["log_mmd"] = f"{logmmd:.{3}f}"
            pbar.set_postfix(display)

    Ws, bs = get_params(opt_state)
    return Ws, bs, S


def learn_single_param(
    muX,
    covX,
    learn_iter=100,
    eta=0.01,
    sampling_alg="pmap",
    n_samples=100,
    n_steps=50,
    use_adam=True,
    rng=random.PRNGKey(42),
):
    # Only used for Section 5.1
    # Learn an Ising model in {-1, 1} with energy E(x) = -theta * \sum_{i<j} x_i x_j
    d = muX.shape[0]
    theta = 0.0
    # Learn theta directly
    opt_init, opt_update, get_params = optimizers.adam(eta)
    opt_state = opt_init(theta)

    def step(opt_stateSrng, it):
        opt_state, S, rng = opt_stateSrng
        theta = get_params(opt_state)

        # Define W, b for data in {-1, 1} and E(x)
        W, b = 0.5 * theta * jnp.ones((d, d)), jnp.zeros((d, 1))
        W = jax.ops.index_update(W, mask, 0)
        # Map to data in {0, 1}
        W, b = stob(W, b)
        # Note: this is equivalent to defining
        # W, b = 4 * theta * jnp.ones((d, d)), -6 * theta * jnp.ones((d, 1))

        rng, rng_input = random.split(rng)
        gW, gb, S = grad(W, b, muX, covX, n_samples, n_steps, sampling_alg, False, rng_input, S)
        gW = jax.ops.index_update(gW, jax.ops.index[mask], 0)

        # Compute gradient for theta
        # dw_ijj / dtheta = 0.5 * 4 = 2
        gtheta = 2 * gW.sum() - 6 * gb.sum()

        opt_state = opt_update(it, - gtheta, opt_state)
        return (opt_state, S, rng), None

    rng, rng_input = random.split(rng)
    S = random.bernoulli(rng_input, p=0.5, shape=(n_samples, d)).astype(jnp.float32)
    mask = jax.ops.index[jnp.eye(d, dtype=bool)]

    opt_state, S, rng = scan(step, (opt_state, S, rng), jnp.arange(learn_iter))[0]
    theta = get_params(opt_state)
    return float(theta)


#####################################
############### Utils ###############
#####################################

@jit
def mean_corr(S):
    # Get moments from data
    mu = S.mean(0, keepdims=True).T
    C = (S.T @ S) / S.shape[0]
    C = (C + C.T) / 2
    C *= 1 - jnp.eye(S.shape[1])
    return mu, C


def stob(W, b):
    # Given the parameters of an Ising model with observations in {-1, 1}
    # and energy E2(x) = - x^T W x - x^T b,
    # returns the equivalent parametrization of an Ising model
    # with observations in {0, 1} and energy E1(x) = - 0.5 * x^T W x - x^T b
    b = 2 * b - 4 * W.sum(1, keepdims=True)
    W = 8 * W
    return W, b


def btos(W, b):
    # Given the parameters of an Ising model with observations in {0, 1}
    # and energy E1(x) = -0.5 * x^T W x - x^T b,
    # returns the equivalent parametrization of an Ising model
    # with observations in {-1, 1} and energy E2(x) = - x^T W x - x^T b
    W = W / 8
    b = (b + 4 * W.sum(1, keepdims=True)) / 2
    return W, b


@jax.partial(jit, static_argnums=(4, 5, 6, 7))  # jit with axis being static
def grad(
    W,
    b,
    muX,
    covX,
    n_samples,
    n_steps,
    sampling_alg,
    signed,
    rng=random.PRNGKey(42),
    S=None,
):
    # Compute the gradient for the different sampling methods
    if signed:
        W, b = stob(W, b)

    # Sample from model
    S, _ = sample(W, b, n_samples, n_steps, sampling_alg, rng, S)
    S1 = S
    if sampling_alg == "gibbs":
        S1 = S[0]
    if signed:
        S1 = 2 * S1 - 1
    mu, C = mean_corr(S1)

    # Gradients = empirical moments - model moments
    gW, gb = covX - C, muX - mu

    if signed:
        return 2 * gW, gb, S
    else:
        return gW, gb, S


#####################################
############# Sampling ##############
#####################################

@jax.partial(jit, static_argnums=(2, 3, 4))  # jit with axis being static
def sample(W, b, n_samples, n_steps, sampling_alg, rng=random.PRNGKey(42), S=None):
    # Sample from an Ising model
    d = W.shape[0]
    pert = None
    if sampling_alg == "pmap" or sampling_alg == "pmap_mplp":
        rng, rng_input = random.split(rng)
        # The difference between two Gumbel follows a logistic
        pert = random.logistic(rng_input, shape=(n_samples,) + b.shape)
        b_pert = b + pert
        mm = min_energy(W, b_pert, n_steps, sampling_alg)
        S = (0 < mm[:, :, 0]).astype(jnp.float32)

    elif sampling_alg == "gibbs":
        if S is None:
            rng, rng_input = random.split(rng)
            S = random.bernoulli(rng_input, p=0.5, shape=(n_samples, d)).astype(jnp.float32)
            S = (S, 0)
        S = gibbs_ising(W, b, S, n_steps, rng)

    elif sampling_alg == "gwg":
        if S is None:
            rng, rng_input = random.split(rng)
            S = random.bernoulli(rng_input, p=0.5, shape=(n_samples, d)).astype(jnp.float32)
        S = gwg_ising(W, b, S, n_steps, rng)

    else:
        assert False, "Unknown sampling method"
    return S, pert


#####################################
############## Gibbs ################
#####################################

@jax.partial(jit, static_argnums=(3,))  # jit with axis being static
def gibbs_ising(W, b, S, n_steps, rng=random.PRNGKey(42)):
    # Vectorization of gibbs sampling for Ising model
    S, i = S
    d = W.shape[0]
    n_samples = S.shape[0]

    def update_gibbs_i(gSrng, i):
        g, S, rng = gSrng

        # Compute probability of switching
        Si = dynamic_slice(S, (0, i), (n_samples, 1))
        gi = dynamic_slice(g, (0, i), (n_samples, 1))
        delta = -(2 * Si - 1) * gi
        threshold = sigmoid(delta)

        # Flip the accepted bits
        rng, rng_input = random.split(rng)
        flip = random.bernoulli(
            rng_input,
            p=threshold,
            shape=(n_samples, 1),
        )
        S = dynamic_update_slice(S, flip - (2 * flip - 1) * Si, (0, i))

        # Update g
        Si = dynamic_slice(S, (0, i), (n_samples, 1))
        Wi = dynamic_slice(W, (i, 0), (1, d))
        delta_g = flip * (2 * Si - 1) * Wi
        g = g.at[:].add(delta_g)
        return (g, S, rng), None

    # We update g in-place to accelerate Gibbs sampling
    g = S @ W.T + b.T
    order = (i + jnp.arange(n_steps)) % d
    g, S, rng = scan(update_gibbs_i, (g, S, rng), order)[0]
    i = (i + n_steps) % d
    return S, i


#####################################
######## Gibbs-with-gradient #########
#####################################

@jax.partial(jit, static_argnums=(3,))  # jit with axis being static
def gwg_ising(W, b, S, n_steps, rng=random.PRNGKey(42)):
    # Gibbs with gradient for Ising model
    rng = random.split(rng, S.shape[0])
    return vmap(gwg_ising1, in_axes=(None, None, 0, None, 0))(W, b, S, n_steps, rng)


@jax.partial(jit, static_argnums=(3,))  # jit with axis being static
def gwg_ising1(W, b, s, n_steps, rng):
    def update_gibbs(gsrng, _):
        g, s, rng = gsrng

        # Sample index
        delta = -(2 * s - 1) * g
        rng, rng_input = random.split(rng)
        i = random.categorical(rng_input, logits=delta / 2)
        s_p = s.at[i].set(1 - s[i])

        # Compute probability of switching
        pm = 2 * s_p[i] - 1
        pmWi = pm * W[i]
        g_p = g + pmWi
        delta_p = -(2 * s_p - 1) * g_p
        p_accept = jnp.exp(
            delta[i] + log_softmax(delta_p / 2)[i] - log_softmax(delta / 2)[i]
        )

        # Flip the accepted bits
        rng, rng_input = random.split(rng)
        flip = random.bernoulli(rng_input, p_accept)
        s = s.at[i].add(flip * pm)
        g = g + flip * pmWi
        return (g, s, rng), None

    g = s @ W + b[:, 0]
    g, s, rng = scan(update_gibbs, (g, s, rng), None, length=n_steps)[0]
    return s


#####################################
############## Max-product ##########
#####################################

# Parallel max-product
@jax.partial(jit, static_argnums=(2, 3))  # jit with axis being static
def min_energy1(W, b, n_steps, sampling_alg="pmap"):
    mp_step = 0.5
    d = W.shape[0]

    # M[i, j] contains the message from node i to node j
    M = jnp.zeros((d, d))
    def update(M, _):
        # Messages are updated in parallel
        mm = (np.ones((1, d)) @ M).T + b
        Min = mm - M.T
        Mnew = maxprop(Min, W)
        if sampling_alg == "pmap_mplp":
            # EMPLP, https://people.csail.mit.edu/tommi/papers/GloJaa_nips07.pdf
            Mnew = 0.5 * Mnew - 0.5 * Min.T

        # Damping
        delta = Mnew - M
        M += mp_step * delta
        return M, None

    M = scan(update, M, None, length=n_steps)[0]
    mm = (np.ones((1, d)) @ M).T + b
    return mm


@jit
def maxprop(mess_in, W):
    # Max-product updates
    mess_out = jnp.maximum(0.0, mess_in + W) - jnp.maximum(0.0, mess_in)
    return mess_out


min_energy = vmap(min_energy1, in_axes=(None, 0, None, None))

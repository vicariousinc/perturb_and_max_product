# Sampling convolutional OR layers  with Gibbs, GWG, PMAP

from jax import grad, jit
from jax import numpy as jnp
from jax import random, vmap
from jax.lax import dynamic_slice, dynamic_update_slice, pad, scan
from jax.nn import log_softmax
from jax.scipy.special import logit as invsigmoid
from functools import partial
from pmap.logical_mpmp import and_bwd, and_fwd, or_fwd


#####################################
########### Convolution #############
#####################################

@jit
def or_layer(S, W):
    n_samples, n_feat, s_height, s_width = S.shape
    n_chan, n_feat, feat_height, feat_width = W.shape
    im_height, im_width = s_height + feat_height - 1, s_width + feat_width - 1

    # Revert the features to have the proper orientations
    Wrev = W[:, :, ::-1, ::-1]

    # Pad the feature locations
    Spad = pad(
        S,
        0.0,
        (
            (0, 0, 0),
            (0, 0, 0),
            (feat_height - 1, feat_height - 1, 0),
            (feat_width - 1, feat_width - 1, 0),
        ),
    )

    # Convolve Spad and W
    def compute_sample(Spad1):
        def compute_pixel(r, c):
            X1 = (
                1
                - dynamic_slice(Spad1, (0, r, c), (n_feat, feat_height, feat_width))
                * Wrev
            ).prod((1, 2, 3))
            return 1 - X1

        compute_cols = vmap(compute_pixel, in_axes=(None, 0), out_axes=1)
        compute_rows_cols = vmap(compute_cols, in_axes=(0, None), out_axes=1)
        return compute_rows_cols(jnp.arange(im_height), jnp.arange(im_width))

    return vmap(compute_sample, in_axes=0, out_axes=0)(Spad)


#####################################
############## Max-product ##########
#####################################

# Parallel max-product
@partial(jit, static_argnums=(3,))  # jit with axis being static
def min_energy(uX, uS, uW, n_steps):
    mp_step = 0.5
    n_samples, n_chan, im_height, im_width = uX.shape
    n_samples, n_feat, s_height, s_width = uS.shape
    n_chan, n_feat, feat_height, feat_width = uW.shape
    assert im_height - feat_height + 1 == s_height
    assert im_width - feat_width + 1 == s_width
    assert uS.shape[1] == uW.shape[1]
    assert uS.shape[0] == uX.shape[0]
    uW = uW[:, :, ::-1, ::-1]

    # Message initialization
    Mout_s = jnp.zeros(
        (n_samples, n_chan, im_height, im_width, n_feat, feat_height, feat_width)
    )
    Mout_w = jnp.zeros(
        (n_samples, n_chan, im_height, im_width, n_feat, feat_height, feat_width)
    )

    def update(M, _):
        # Incoming messages
        Mout_s, Mout_w = M
        Min_s, _ = incoming_s(Mout_s, uS)
        Min_w = Mout_w.sum((0, 2, 3), keepdims=True) + uW[None, :, None, None] - Mout_w

        # Max product updates
        Mnew_s, Mnew_w = maxprop(Min_s, Min_w, uX)
        delta_s, delta_w = Mnew_s - Mout_s, Mnew_w - Mout_w

        # Damping
        Mout_s += mp_step * delta_s
        Mout_w += mp_step * delta_w

        return (
            (Mout_s, Mout_w),
            jnp.maximum(jnp.abs(delta_w).max(), jnp.abs(delta_s).max()),
        )

    (Mout_s, Mout_w), convergence = scan(update, (Mout_s, Mout_w), None, length=n_steps)

    # Compute beliefs
    _, bel_s = incoming_s(Mout_s, uS)
    bel_w = Mout_w.sum((0, 2, 3)) + uW
    return bel_s, bel_w[:, :, ::-1, ::-1], convergence


@jit
def maxprop(Min_s, Min_w, uX):
    assert Min_s.shape == Min_w.shape
    _, _, _, _, n_feat, feat_height, feat_width = Min_w.shape

    # AND pool from features and locations to channels
    int_down = and_bwd(Min_s.ravel(), Min_w.ravel())

    # OR poool from image to channels
    int_up = or_fwd(
        int_down.reshape(-1, n_feat * feat_height * feat_width), uX.reshape(-1)
    ).ravel()
    # Mout_x = or_bwd(int_down.reshape(-1, n_feat * feat_height * feat_width))

    # AND pool from channels to features and locatioons
    Mout_s, Mout_w = and_fwd(Min_s.ravel(), Min_w.ravel(), int_up)
    return (
        Mout_s.reshape(Min_s.shape),
        Mout_w.reshape(Min_w.shape),
        # Mout_x.reshape(uX.shape),
    )


@jit
def incoming_s(Mout_s, uS):
    # Compute incoming messages for features locations
    # A feature location activates different feature coordinates at different image pixels
    (
        n_samples,
        n_chan,
        im_height,
        im_width,
        n_feat,
        feat_height,
        feat_width,
    ) = Mout_s.shape

    assert uS.shape == (
        n_samples,
        n_feat,
        im_height - feat_height + 1,
        im_width - feat_width + 1,
    )
    # Pad the beliefs
    bel_s = pad(
        uS,
        -jnp.inf,
        (
            (0, 0, 0),
            (0, 0, 0),
            (feat_height - 1, feat_height - 1, 0),
            (feat_width - 1, feat_width - 1, 0),
        ),
    ).transpose((0, 2, 3, 1))

    # Update beliefs
    def update_bel(r, c):
        update = dynamic_slice(
            bel_s, (0, r, c, 0), (n_samples, im_height, im_width, n_feat)
        ) + Mout_s[:, :, :, :, :, r, c].sum(1)
        return dynamic_update_slice(bel_s, update, (0, r, c, 0))

    # Update message to OR pools
    def update_Min(bel_s, r, c):
        return (
            dynamic_slice(
                bel_s, (0, r, c, 0), (n_samples, im_height, im_width, n_feat)
            )[:, None]
            - Mout_s[:, :, :, :, :, r, c]
        )

    # A feature locations at (r, c)
    bel_s = vmap(vmap(update_bel, (None, 0)), (0, None))(
        jnp.arange(feat_height), jnp.arange(feat_width)
    ).sum((0, 1))

    Min_s = vmap(vmap(update_Min, (None, None, 0)), (None, 0, None))(
        bel_s, jnp.arange(feat_height), jnp.arange(feat_width)
    ).transpose((2, 3, 4, 5, 6, 0, 1))
    return (
        Min_s,
        bel_s.transpose((0, 3, 1, 2))[
            :,
            :,
            feat_height - 1 : im_height,
            feat_width - 1 : im_width,
        ],
    )


#####################################
######## Gibbs-with-gradient ########
#####################################

@partial(jit, static_argnums=(2, 3))  # jit with axis being static
def _gwg_co(X, sw, Wshape, n_steps, rng):
    def delta(sw):
        return -(2 * sw - 1) * glogprob(X, sw, Wshape)

    def update_gibbs(SWrng, _):
        sw, rng = SWrng
        rng, rng_input = random.split(rng)
        delta_sw = delta(sw)
        # delta_sw = delta_sw.at[:1280].set(-jnp.inf)
        i = random.categorical(rng_input, logits=delta_sw / 2)
        sw_p = sw.at[i].set(1 - sw[i])
        logprob_sw = logprob_flat(X, sw, Wshape)
        p_accept = jnp.exp(
            logprob_flat(X, sw_p, Wshape)
            - logprob_sw
            + log_softmax(delta(sw_p) / 2)[i]
            - log_softmax(delta_sw / 2)[i]
        )
        rng, rng_input = random.split(rng)
        flip = random.bernoulli(rng_input, p_accept)
        sw = sw.at[i].add(flip * (1 - 2 * sw[i]))
        return (sw, rng), logprob_sw

    (sw, rng), convergence = scan(update_gibbs, (sw, rng), None, length=n_steps)
    return sw, convergence


@partial(jit, static_argnums=(3,))
def gwg_co(X, S, W, n_steps, rng=random.PRNGKey(42)):
    sw = jnp.hstack((S.ravel(), W.ravel()))
    sw, convergence = _gwg_co(X, sw, W.shape, n_steps, rng)
    S, W = sw_to_S_W(X, sw, W.shape)
    return S, W, convergence


# 1-(1-p1*w1)*(1-p2*w2)*(1-p3*w3)
# log p(y=1) = log(1-sum_i log(1-exp(log(si)+log(wi)))  ))
# log p(y=0) = sum_i log(1-exp(log(si)+log(wi)))


@partial(jit, static_argnums=(2,))  # jit with axis being static
def sw_to_S_W(X, sw, Wshape):
    n_chan, n_feat, feat_height, feat_width = Wshape
    n_samples, n_chan, im_height, im_width = X.shape
    s_height, s_width = im_height - feat_height + 1, im_width - feat_width + 1
    slength = n_samples * n_feat * s_height * s_width
    S = dynamic_slice(sw, (0,), (slength,)).reshape(
        n_samples, n_feat, s_height, s_width
    )
    W = sw[slength:].reshape(Wshape)
    return S, W


@partial(jit, static_argnums=(2,))  # jit with axis being static
def logprob_flat(X, sw, Wshape):
    pW = 0.15
    pS = 0.05
    # pe = 0.00669283
    S, W = sw_to_S_W(X, sw, Wshape)
    Xhat = or_layer(S, W)

    ipW = invsigmoid(pW)
    ipS = invsigmoid(pS)
    ipX = (2 * X - 1) * 1000  # -invsigmoid(pe)

    return (
        (ipX * Xhat).sum((1, 2, 3)).mean()
        + (ipW * W).sum() / X.shape[0]
        + (ipS * S).sum((1, 2, 3)).mean()
    )


def logprob(X, S, W):
    sw = jnp.hstack((S.ravel(), W.ravel()))
    return logprob_flat(X, sw, W.shape)


glogprob = grad(logprob_flat, 1)

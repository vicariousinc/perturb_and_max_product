# Messages updates for OR and AND pools

from jax import jit
from jax import numpy as jnp
from jax import vmap


@jit
def _or_fwd(t_in, b_in):  # 1D version
    # Message from bottom of OR pool to top
    # OR pool is of form:
    # t1   t2   t3   t4
    # |    |    |    |
    #  \   |    |    /
    #    \  \  /   /
    #         F
    #         |
    #         b
    # t_in: messages from the top variables to the factor F
    # b_in: message from the bottom variable to the factor F
    assert len(t_in.shape) == 1
    assert len(b_in.shape) == 0
    assert len(t_in) > 0, "logical gate with zero inputs"
    if len(t_in) == 1:
        return jnp.array(b_in)

    t_in_pos = jnp.maximum(0.0, t_in)
    min_left = b_in + t_in_pos.sum() - t_in_pos

    # We need the first and second largest message
    f_loc = t_in.argmax()
    s_loc = t_in.at[f_loc].set(-jnp.inf).argmax()

    # Messages to all variables but the winner
    t_out = jnp.minimum(min_left, t_in_pos[f_loc] - t_in[f_loc])

    # Message to the winner
    t_out = t_out.at[f_loc].set(
        jnp.minimum(min_left[f_loc], t_in_pos[s_loc] - t_in[s_loc])
    )
    return t_out


or_fwd = vmap(_or_fwd, in_axes=(0, 0))  # 2D version


@jit
def _or_bwd(t_in):  # 1D version
    # Message from top of OR pool to bottom
    # OR pool is of form:
    # t1   t2   t3   t4
    # |    |    |    |
    #  \   |    |    /
    #    \  \  /   /
    #         F
    #         |
    #         b
    # t_in: messages from the top variables to the factor F
    assert len(t_in.shape) == 1
    assert len(t_in) > 0, "logical gate with zero inputs"
    t_out = jnp.maximum(0.0, t_in).sum() + jnp.minimum(0.0, t_in.max())
    return t_out


or_bwd = vmap(_or_bwd)  # 2D version


@jit
def _and_fwd(t_in0, t_in1, b_in):  # 1D version
    # Message from bottom of AND pool to top
    # AND pool is of form:
    # t1     t2
    #   \   /
    #     F
    #     |
    #     b
    # t_in0, t_in1: messages from the top variables to the factor F
    # b_in: message from bottom variable to the factor F
    assert t_in0.shape == t_in0.shape == b_in.shape == ()
    return (
        jnp.maximum(0, t_in1 + b_in) - jnp.maximum(0, t_in1),
        jnp.maximum(0, t_in0 + b_in) - jnp.maximum(0, t_in0),
    )


and_fwd = vmap(_and_fwd, in_axes=(0, 0, 0))  # 2D version


@jit
def _and_bwd(t_in0, t_in1):  # 1D version
    # Message from top of AND pool to bottom
    # AND pool is of form:
    # t1     t2
    #   \   /
    #     F
    #     |
    #     b
    # t_in0, t_in1: messages from the top variables to the factor F
    # b_in: message from bottom variable to the factor F
    assert t_in0.shape == t_in0.shape == ()
    b_out = jnp.minimum(t_in0 + t_in1, jnp.minimum(t_in0, t_in1))
    return b_out


and_bwd = vmap(_and_bwd, in_axes=(0, 0))  # 2D version

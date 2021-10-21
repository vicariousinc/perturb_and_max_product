import os
import numpy as np
import cvxpy as cp
import ecos
from tqdm import trange
from scipy.sparse import save_npz, load_npz
from joblib import Parallel, delayed, cpu_count
from pmap.ising_modeling import stob


#####################################
############## Learning #############
#####################################

def learn_lp(
    muX,
    covX,
    learn_iter=1000,
    n_samples=100,
    use_adam=True,
    eta=0.01,
    signed=False,
    seed=0,
    n_jobs=None
):
    np.random.seed(seed)
    d = muX.shape[0]
    W, b = np.zeros((d, d)), np.zeros((d, 1))

    # Adam params
    assert use_adam, "Optimizer not implemented!"
    if use_adam:
        v_b, sqrt_b = np.zeros_like(b), np.zeros_like(b)
        v_W, sqrt_W = np.zeros_like(W), np.zeros_like(W)

    pbar = trange(learn_iter)
    for it in pbar:
        # Gradient step
        grad_W, grad_b, S, _ = grad_lp(W, b, muX, covX, n_samples, signed, n_jobs=n_jobs)
        if use_adam:
            b = adam(b, v_b, sqrt_b, grad_b, eta, it + 1)
            W = adam(W, v_W, sqrt_W, grad_W, eta, it + 1)
    return W, b, S


def learn_single_param_lp(
    muX,
    covX,
    learn_iter=100,
    eta=0.1,
    n_samples=100,
    seed=0,
    n_jobs=None
):
    np.random.seed(seed)

    # Only used for Section 5.1
    # Learn an Ising model in {-1, 1} with energy E(x) = -theta * \sum_{i<j} x_i x_j
    d = muX.shape[0]
    theta = np.array([0.0])
    v_theta, sqrt_theta = np.zeros_like(theta), np.zeros_like(theta)

    for it in range(learn_iter):
        W, b = .5 * theta * np.ones((d, d)), np.zeros((d, 1))
        np.fill_diagonal(W, 0)
        # Map to data in {0, 1} and E2
        W, b = stob(W, b)

        # Gradient step
        grad_W, grad_b, S, objvals = grad_lp(W, b, muX, covX, n_samples, signed=False, n_jobs=n_jobs)
        gtheta = np.array([2 * grad_W.sum() - 6 * grad_b.sum()])

        theta = adam(theta, v_theta, sqrt_theta, gtheta, eta, it + 1)
    return float(theta)


#####################################
############## Sampling #############
#####################################

def grad_lp(
    W,
    b,
    muX,
    covX,
    n_samples,
    signed,
    n_jobs=None
):
    S, objvals = sample_lp(W, b, n_samples, n_jobs=n_jobs)
    mu, C = np_mean_corr(S)
    if signed:
        S = 2 * S - 1

    # Gradients = empirical moments - model moments
    grad_W, grad_b = covX - C, muX - mu

    if signed:
        grad_W = 2 * grad_W
    return grad_W, grad_b, S, objvals


def np_mean_corr(S):
    # Get moments from data
    mu = S.mean(0, keepdims=True).T
    C = (S.T @ S) / S.shape[0]
    C = (C + C.T) / 2
    C *= 1 - np.eye(S.shape[1])
    return mu, C


def adam(W, vW, sqrW, gW, lr, t):
    # https://gluon.mxnet.io/chapter06_optimization/adam-scratch.html
    # Same parameters as JAX https://objax.readthedocs.io/en/latest/objax/optimizer.html
    beta1 = 0.9
    beta2 = 0.999
    eps_stable = 1e-8

    vW[:] = beta1 * vW + (1.0 - beta1) * gW
    sqrW[:] = beta2 * sqrW + (1.0 - beta2) * np.square(gW)

    v_bias_corr = vW / (1.0 - beta1 ** t)
    sqr_bias_corr = sqrW / (1.0 - beta2 ** t)

    div = lr * v_bias_corr / (np.sqrt(sqr_bias_corr) + eps_stable)
    return W + div


def sample_lp(W, b, n_samples, n_jobs=None):
    pert = np.random.logistic(size=(n_samples,) + b.shape)
    b_pert = b + pert
    S, objvals = solve_lp(W, b_pert, n_jobs=n_jobs)
    # Samples must have integer values
    S = S.round()
    return S, objvals


#####################################
############# LP solving ############
#####################################

def solve_lp(W, b_pert, n_jobs=None):
    dim_v = b_pert.shape[1]
    list_b_pert = [b for b in b_pert]

    model_folder = "/tmp/model_ising_ecos_{}/".format(dim_v)
    if not os.path.exists(model_folder):
        model_G, model_h = create_ising_ecos_model(dim_v, model_folder)
    else:
        model_G = load_npz(model_folder + '/G.npz')
        model_h = np.load(model_folder + '/h.npy')

    # Solve LPs in parallel
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    results = Parallel(n_jobs=n_jobs)(
        delayed(solve_ising_ecos)(W=W, bv=bv, model_G=model_G, model_h=model_h) for bv in list_b_pert
    )
    vs = [res[0] for res in results]
    objvals = [res[2] for res in results]
    return np.array(vs), np.array(objvals)


def create_ising_ecos_model(dim_v, model_folder):
    # See https://people.csail.mit.edu/dsontag/papers/sontag_phd_thesis.pdf
    v = cp.Variable((dim_v, 1))
    z = cp.Variable((dim_v, dim_v))

    constraints = [
        v @ np.ones((1, dim_v)) + np.ones((dim_v, 1) )@ v.T <= 1 + z,
        z <= v @ np.ones((1, dim_v)),
        z <= np.ones((dim_v, 1) )@ v.T,
        0 <= z,
        v <=1,
    ]

    # Variables
    W_v, b_v = np.zeros((dim_v, dim_v)), np.zeros((dim_v, 1))

    prob = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(b_v, v)) + 0.5 * cp.sum(cp.multiply(W_v, z))),
        constraints,
    )
    data = prob.get_problem_data(cp.ECOS)[0]
    model_G, model_h = data['G'], data['h']

    os.mkdir(model_folder)
    save_npz(model_folder + '/G.npz', model_G)
    np.save(model_folder + '/h.npy', model_h)
    print("Ecos model created for dimension {}".format(dim_v))
    return model_G, model_h


def solve_ising_ecos(W, bv, model_G, model_h):
    dim_v = bv.shape[0]
    assert W.shape == (dim_v, dim_v)
    theta = np.hstack((bv.flatten(), 0.5 * W.T.flatten()))

    # Solve the LP, which is expressed as a minimization problem
    vals = ecos.solve(- theta, model_G, model_h, dims={'l':model_G.shape[0]}, verbose=False)['x']

    v = vals[:dim_v]
    z = vals[dim_v:].reshape(dim_v, dim_v).T
    objval = - (theta * vals).sum()
    return v, z, objval

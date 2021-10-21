import os
import numpy as np
import cvxpy as cp
import ecos
from tqdm import trange
from scipy.sparse import save_npz, load_npz
from joblib import Parallel, delayed, cpu_count
from jax.nn import sigmoid, softplus

from .ising_modeling_lp import adam

#####################################
############## Learning #############
#####################################

def learn_lp(
    X,
    nh,
    mb_size=100,
    learn_iter=100,
    n_samples=100,
    eta=0.1,
    use_adam=True,
    seed=0
):
    np.random.seed(seed)
    nv = X.shape[1]
    W, bv, bh = np.random.normal(0, 0.01, (nh, nv)), np.random.normal(0, 1, (1, nv)), np.random.normal(0, 1, (1, nh))

    Xmb = X[np.random.permutation(X.shape[0]), :].reshape(-1, mb_size, nv)

    # Adam params
    assert use_adam, "Optimizer not implemented!"
    if use_adam:
        v_W, sqrt_W = np.zeros_like(W), np.zeros_like(W)
        v_bv, sqrt_bv = np.zeros_like(bv), np.zeros_like(bv)
        v_bh, sqrt_bh = np.zeros_like(bh), np.zeros_like(bh)

    convergence = []
    display = {}
    pbar = trange(learn_iter)
    for it in pbar:
        # Gradient step
        gW, gbv, gbh, logZmodel, logZdata, S = grad_lp(W, bv, bh, Xmb[it % Xmb.shape[0]], n_samples)

        log_lik = logZdata.mean() - logZmodel.mean()
        convergence.append(log_lik)
        display["log_lik"] = log_lik

        if use_adam:
            W = adam(W, v_W, sqrt_W, gW, eta, it + 1)
            bv = adam(bv, v_bv, sqrt_bv, gbv, eta, it + 1)
            bh = adam(bh, v_bh, sqrt_bh, gbh, eta, it + 1)
        pbar.set_postfix(display)

    return W, bv, bh, convergence, S


#####################################
############## Sampling #############
#####################################

def grad_lp(W, bv, bh, X, n_samples):
    # First term of the gradient
    zdata_W, zdata_bv, zdata_bh = grad_from_samples_np(W, bh, X)

    # Marginal disitribution of the visible
    # https://christian-igel.github.io/paper/AItRBM-proof.pdf Eq 20
    logitH = X @ W.T + bh
    logZdata = (X * bv).sum(1) + softplus(logitH).sum(1)

    # Sample from model
    # logZmodel is currently 0
    S, logZmodel = sample_lp(W, bv, bh, n_samples)

    # Second term of the gradient
    zmodel_W, zmodel_bv, zmodel_bh = grad_from_samples_np(W, bh, S)

    # Gradient
    gW = zdata_W.mean(0) - zmodel_W.mean(0)
    gbv = zdata_bv.mean(0) - zmodel_bv.mean(0)
    gbh = zdata_bh.mean(0) - zmodel_bh.mean(0)

    return gW, gbv, gbh, logZmodel, logZdata, S


def sample_lp(W, bv, bh, n_samples):
    nv = bv.shape[1]
    nh = bh.shape[1]

    # Samples
    v_pert = np.random.logistic(size=(n_samples, nv))
    bv_pert = bv + v_pert
    h_pert = np.random.logistic(size=(n_samples, nh))
    bh_pert = bh + h_pert
    S, logZmodel = solve_lp(W, bv_pert, bh_pert)
    # LP solutions are fractional, but samples must have integer values
    S = S.round()
    return S, logZmodel


def grad_from_samples_np(W, bh, X):
    n_samples = X.shape[0]

    # https://christian-igel.github.io/paper/AItRBM-proof.pdf Eq 28, 32, 33
    H = sigmoid(X.dot(W.T) + bh.reshape(-1))
    zh, zv = (
        H.reshape(n_samples, 1, -1),
        X.reshape(n_samples, 1, -1),
    )
    Z = zh.transpose((0, 2, 1)) * zv
    return Z, zv, zh


#####################################
######### LP solving ################
#####################################

def solve_lp(W, bv_pert, bh_pert):
    dim_v = bv_pert.shape[1]
    dim_h = bh_pert.shape[1]
    list_b_perts = [(bv, bh) for bv, bh in zip(bv_pert, bh_pert)]

    model_folder = "/tmp/model_rbm_ecos_{}_{}/".format(dim_v, dim_h)
    if not os.path.exists(model_folder):
        model_G, model_h = create_rbm_ecos_model(dim_v, dim_h, model_folder)
    else:
        model_G = load_npz(model_folder + '/G.npz')
        model_h = np.load(model_folder + '/h.npy')

    results = Parallel(n_jobs=cpu_count() -1)(
        delayed(solve_rbm_ecos)(W=W, b_perts=b_perts, model_G=model_G, model_h=model_h) for b_perts in list_b_perts
    )

    vs = [res[0] for res in results]
    logZs = [res[3] for res in results]
    return np.array(vs), np.array(logZs)


def create_rbm_ecos_model(dim_v, dim_h, model_folder):
    # Variables
    v = cp.Variable((1, dim_v))
    h = cp.Variable((dim_h, 1))
    z = cp.Variable((dim_h, dim_v))

    # Constraints
    constraints = [
        h @ np.ones((1, dim_v)) + np.ones((dim_h, 1) )@ v <= 1 + z,
        z <= h @ np.ones((1, dim_v)),
        z <= np.ones((dim_h, 1)) @ v,
        0 <= z,
        h <= 1,
        v <=1,
    ]

    # Objective value
    W, bv, bh = np.ones((dim_h, dim_v)), np.ones((1, dim_v)), np.ones((dim_h, 1))

    prob = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(bv, v)) + cp.sum(cp.multiply(bh, h)) + cp.sum(cp.multiply(W, z))),
        constraints,
    )
    data = prob.get_problem_data(cp.ECOS)[0]
    G, h = data['G'], data['h']

    os.mkdir(model_folder)
    save_npz(model_folder + '/G.npz', G)
    np.save(model_folder + '/h.npy', h)
    print("Ecos model created for dimensions {} {}".format(dim_v, dim_h))
    return G, h


def solve_rbm_ecos(W, b_perts, model_G, model_h):
    bv, bh = b_perts
    dim_v = bv.shape[0]
    dim_h = bh.shape[0]
    assert W.shape == (dim_h, dim_v)

    theta = np.hstack((bv.flatten(), bh.flatten(), W.T.flatten()))

    # Solution
    vals = ecos.solve(- theta, model_G, model_h, dims={'l':model_G.shape[0]}, verbose=False)['x']

    v = vals[:dim_v]
    h = vals[dim_v: dim_h + dim_v]
    z = vals[dim_h + dim_v:].reshape(dim_v, dim_h).T

    # Upper-bound of the log partition function using Gumbel noise
    # https://arxiv.org/pdf/1206.6410.pdf Corollary 1
    V = v.round()
    H = h.round()
    logZ = V.dot(W.T).dot(H) + V.dot(bv) + H.dot(bh)
    return v, h, z, logZ

import os

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import tensorly as tl
from tensorly.tenalg import khatri_rao as kr
from tensorly.cp_tensor import unfolding_dot_khatri_rao
from tensorly.decomposition import parafac

from utils import reindex, tensor_unfold, tensor_refold

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

data_aminoacids = scipy.io.loadmat("data/amino.mat")['X']

# PARAFAC
frob_cpd = []
param_cpd = []
norm = np.linalg.norm(data_aminoacids.ravel())
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    weights, factors = parafac(
        tl.tensor(data_aminoacids, dtype=tl.float32),
        rank=i,
        normalize_factors=False,
        init='random',
        tol=1e-3,
        n_iter_max=100
    )
    error = np.linalg.norm(data_aminoacids.ravel() - tl.kruskal_to_tensor((weights, factors)).ravel())
    frob_cpd.append(100*error/norm)
    param_cpd.append(np.sum(data_aminoacids.shape)*i)

# MRLR-RES-1
frob_mrlr_res_1 = []
params_mrlr_res_1 = []
norm = np.linalg.norm(data_aminoacids.ravel())
for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    rank = i
    rank_2 = 4

    X_2, X_2_recovery_map = tensor_unfold(data_aminoacids, [[1], [0, 2]])
    X_3 = data_aminoacids

    X_target_2 = X_2
    weights, factors_2_hat = parafac(
        tl.tensor(X_target_2, dtype=tl.float32),
        rank=rank_2,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_2_hat = tl.kruskal_to_tensor((weights, factors_2_hat))

    R_2 = tensor_refold(X_2_hat, X_2_recovery_map, data_aminoacids.shape)
    X_target_3 = X_3 - R_2
    weights, factors_3_hat = parafac(
        tl.tensor(X_target_3, dtype=tl.float32),
        rank=rank,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_3_hat = tl.kruskal_to_tensor((weights, factors_3_hat))

    X_hat = R_2 + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank

    error = np.linalg.norm(data_aminoacids.ravel() - X_hat.ravel())
    frob_mrlr_res_1.append(100*error/norm)
    params_mrlr_res_1.append(n_elements)

# MRLR-RES-2
frob_mrlr_res_2 = []
params_mrlr_res_2 = []
norm = np.linalg.norm(data_aminoacids.ravel())
for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    rank = i
    rank_2 = 4

    X_2, X_2_recovery_map = tensor_unfold(data_aminoacids, [[2], [0, 1]])
    X_3 = data_aminoacids

    X_target_2 = X_2
    weights, factors_2_hat = parafac(
        tl.tensor(X_target_2, dtype=tl.float32),
        rank=rank_2,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_2_hat = tl.kruskal_to_tensor((weights, factors_2_hat))

    R_2 = tensor_refold(X_2_hat, X_2_recovery_map, data_aminoacids.shape)
    X_target_3 = X_3 - R_2
    weights, factors_3_hat = parafac(
        tl.tensor(X_target_3, dtype=tl.float32),
        rank=rank,
        normalize_factors=False,
        init='random',
        tol=1e-3
        )
    X_3_hat = tl.kruskal_to_tensor((weights, factors_3_hat))

    X_hat = R_2 + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank

    error = np.linalg.norm(data_aminoacids.ravel() - X_hat.ravel())
    frob_mrlr_res_2.append(100*error/norm)
    params_mrlr_res_2.append(n_elements)

# MRLR-RES-INV-1
frob_mrlr_invres_1 = []
params_mrlr_invres_1 = []
norm = np.linalg.norm(data_aminoacids.ravel())
for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    rank = i
    rank_2 = 4

    X_2, X_2_recovery_map = tensor_unfold(data_aminoacids, [[1], [0, 2]])
    X_3 = data_aminoacids

    X_target_3 = X_3
    weights, factors_3_hat = parafac(
        tl.tensor(X_target_3, dtype=tl.float32),
        rank=rank,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_3_hat = tl.kruskal_to_tensor((weights, factors_3_hat))

    R_3, R_3_recovery_map = tensor_unfold(X_3_hat, [[1], [0, 2]])
    X_target_2 = X_2 - R_3
    weights, factors_2_hat = parafac(
        tl.tensor(X_target_2, dtype=tl.float32),
        rank=rank_2,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_2_hat = tl.kruskal_to_tensor((weights, factors_2_hat))

    X_hat = tensor_refold(X_2_hat, X_2_recovery_map, data_aminoacids.shape) + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank

    error = np.linalg.norm(data_aminoacids.ravel() - X_hat.ravel())
    frob_mrlr_invres_1.append(100*error/norm)
    params_mrlr_invres_1.append(n_elements)

# MRLR-RES-INV-2
frob_mrlr_invres_2 = []
params_mrlr_invres_2 = []
norm = np.linalg.norm(data_aminoacids.ravel())
for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    rank = i
    rank_2 = 4

    X_2, X_2_recovery_map = tensor_unfold(data_aminoacids, [[2], [0, 1]])
    X_3 = data_aminoacids

    X_target_3 = X_3
    weights, factors_3_hat = parafac(
        tl.tensor(X_target_3, dtype=tl.float32),
        rank=rank,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_3_hat = tl.kruskal_to_tensor((weights, factors_3_hat))

    R_3, R_3_recovery_map = tensor_unfold(X_3_hat, [[2], [0, 1]])
    X_target_2 = X_2 - R_3
    weights, factors_2_hat = parafac(
        tl.tensor(X_target_2, dtype=tl.float32),
        rank=rank_2,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_2_hat = tl.kruskal_to_tensor((weights, factors_2_hat))

    X_hat = tensor_refold(X_2_hat, X_2_recovery_map, data_aminoacids.shape) + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank

    error = np.linalg.norm(data_aminoacids.ravel() - X_hat.ravel())
    frob_mrlr_invres_2.append(100*error/norm)
    params_mrlr_invres_2.append(n_elements)

# PLOT
pparam = dict(ylabel='$NFE (\%)$', xlabel='$\# Parameters$')
with plt.style.context(['science'], ['ieee']):
    fig, ax = plt.subplots()
    plt.grid()
    ax.plot(param_cpd, frob_cpd, label="PARAFAC", marker='o', c='k')
    ax.plot(params_mrlr_res_1, frob_mrlr_res_1, label="MRLR-res-1", marker='D', c='orange')
    ax.plot(params_mrlr_res_2, frob_mrlr_res_2, label="MRLR-res-2", marker='x', c='blue')
    ax.plot(params_mrlr_invres_1, frob_mrlr_invres_1, label="MRLR-res-1-reverse", marker='s', c='r')
    ax.plot(params_mrlr_invres_2, frob_mrlr_invres_2, label="MRLR-res-2-reverse", marker='p', c='g')
    ax.legend()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0., 1.5)
    ax.set_xlim(1500., 30000)
    fig.savefig('results/fig_1.jpg', dpi=300)
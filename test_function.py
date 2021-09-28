import os

import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl

from tensorly.tenalg import khatri_rao as kr
from tensorly.cp_tensor import unfolding_dot_khatri_rao
from tensorly.decomposition import parafac

from utils import reindex, tensor_unfold, tensor_refold

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

grid = np.arange(-5, 5, 0.1)
length = len(grid)
tensor = np.zeros((length, length, length))
for i_x, x in enumerate(grid):
    for i_y, y in enumerate(grid):
        for i_z, z in enumerate(grid):
            tensor[i_x, i_y, i_z] = (x**2 + y**2)/np.exp(np.abs(y + z))

# PARAFAC
frob_cpd = []
param_cpd = []
norm = np.linalg.norm(tensor)
for i in range(1, 200, 10):
    weights, factors = parafac(
        tl.tensor(tensor, dtype=tl.float32),
        rank=i,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    tensor_hat = tl.kruskal_to_tensor((weights, factors))
    error = np.linalg.norm(tensor.ravel() - tensor_hat.ravel())
    frob_cpd.append(100*error/norm)
    param_cpd.append(np.sum(tensor.shape)*i)

# MRLR
frob_mtd = []
param_mtd = []
norm = np.linalg.norm(tensor)
for i in range(1, 100, 10):
    rank_2 = 1
    rank_3 = i

    X_2, X_2_recovery_map = tensor_unfold(tensor, [[0], [1, 2]])
    X_3 = tensor

    X_target_2 = X_2
    weights, factors_2_hat = parafac(
        tl.tensor(X_target_2, dtype=tl.float32),
        rank=rank_2,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_2_hat = tl.kruskal_to_tensor((weights, factors_2_hat))

    R_2 = tensor_refold(X_2_hat, X_2_recovery_map, tensor.shape)
    X_target_3 = X_3 - R_2
    weights, factors_3_hat = parafac(
        tl.tensor(X_target_3, dtype=tl.float32),
        rank=rank_3,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_3_hat = tl.kruskal_to_tensor((weights, factors_3_hat))

    X_hat = R_2 + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2 \
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank_3

    error = np.linalg.norm(tensor.ravel() - X_hat.ravel())
    frob_mtd.append(100*error/norm)
    param_mtd.append(n_elements)

# PLOT
pparam = dict(ylabel='$NFE (\%)$', xlabel='$\# Parameters$')
with plt.style.context(['science'], ['ieee']):
    fig, ax = plt.subplots()
    plt.grid()
    ax.plot(param_cpd, frob_cpd, label="PARAFAC", marker='o', c='k')
    ax.plot(param_mtd, frob_mtd, label="MRLR", marker='D', c='orange')
    ax.legend()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0, 5)
    ax.set_xlim(7000, 40000)
    fig.savefig('results/fig_3.jpg', dpi=300)
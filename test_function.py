import os

import numpy as np
import matplotlib.pyplot as plt

from utils import reindex, tensor_unfold, tensor_refold, parafac_decomposition

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

grid = np.arange(-5, 5, 0.1)
length = len(grid)
tensor = np.zeros((length, length, length))
for i_x, x in enumerate(grid):
    for i_y, y in enumerate(grid):
        for i_z, z in enumerate(grid):
            tensor[i_x, i_y, i_z] = (x**2 + y**2)/np.exp(np.abs(y + z))
norm = np.linalg.norm(tensor)
ranks = range(1, 200, 10)

# PARAFAC
nfe_parafac = list()
params_parafac = list()
for i in ranks:
    tensor_hat = parafac_decomposition(tensor, i)
    error = np.linalg.norm(tensor.ravel() - tensor_hat.ravel())
    nfe_parafac.append(100*error/norm)
    params_parafac.append(np.sum(tensor.shape)*i)


# MRLR
nfe_mrlr = []
params_mrlr = []
partition = [[0], [1, 2]]
rank_2 = 1
for i in ranks:
    X_2, X_2_recovery_map = tensor_unfold(tensor, partition)
    X_2_hat = parafac_decomposition(X_2, rank_2)
    R_2 = tensor_refold(X_2_hat, X_2_recovery_map, tensor.shape)

    X_3_hat = parafac_decomposition(tensor - R_2, i)

    X_hat = R_2 + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2 \
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*i

    error = np.linalg.norm(tensor.ravel() - X_hat.ravel())
    nfe_mrlr.append(100*error/norm)
    params_mrlr.append(n_elements)

# PLOT
pparam = dict(ylabel='$NFE (\%)$', xlabel='$\# Parameters$')
with plt.style.context(['science'], ['ieee']):
    fig, ax = plt.subplots()
    plt.grid()
    ax.plot(params_parafac, nfe_parafac, label="PARAFAC", marker='o', c='k')
    ax.plot(params_mrlr, nfe_mrlr, label="MRLR", marker='D', c='orange')
    ax.legend()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0, 5)
    ax.set_xlim(7000, 40000)
    fig.savefig('results/fig_3.jpg', dpi=300)
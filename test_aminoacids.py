import os

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from utils import reindex, tensor_unfold, tensor_refold, parafac_decomposition

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

data = scipy.io.loadmat("data/amino.mat")['X']
norm = np.linalg.norm(data.ravel())
ranks = range(10, 110, 10)

# PARAFAC
nfe_parafac = list()
params_parafac = list()
for i in ranks:
    tensor_hat = parafac_decomposition(data, i)
    error = np.linalg.norm(data.ravel() - tensor_hat.ravel())
    nfe_parafac.append(100*error/norm)
    params_parafac.append(np.sum(data.shape)*i)

# MRLR-RES-1
partition = [[1], [0, 2]]
rank_2 = 4
nfe_mrlr_res_1 = []
params_mrlr_res_1 = []
for i in ranks:
    X_2, X_2_recovery_map = tensor_unfold(data, partition)
    X_2_hat = parafac_decomposition(X_2, rank_2)
    R_2 = tensor_refold(X_2_hat, X_2_recovery_map, data.shape)

    X_3_hat = parafac_decomposition(data - R_2, i)
    X_hat = R_2 + X_3_hat

    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*i

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    nfe_mrlr_res_1.append(100*error/norm)
    params_mrlr_res_1.append(n_elements)

# MRLR-RES-2
partition = [[2], [0, 1]]
rank_2 = 4
nfe_mrlr_res_2 = []
params_mrlr_res_2 = []
for i in ranks:
    X_2, X_2_recovery_map = tensor_unfold(data, partition)
    X_2_hat = parafac_decomposition(X_2, rank_2)
    R_2 = tensor_refold(X_2_hat, X_2_recovery_map, data.shape)

    X_3_hat = parafac_decomposition(data - R_2, i)
    X_hat = R_2 + X_3_hat

    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*i

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    nfe_mrlr_res_2.append(100*error/norm)
    params_mrlr_res_2.append(n_elements)

# MRLR-RES-INV-1
partition = [[1], [0, 2]]
nfe_mrlr_invres_1 = []
params_mrlr_invres_1 = []
for i in ranks:
    X_3_hat = parafac_decomposition(data, i)
    R_3, R_3_recovery_map = tensor_unfold(X_3_hat, partition)

    X_2, X_2_recovery_map = tensor_unfold(data, partition)
    X_2_hat = parafac_decomposition(X_2 - R_3, rank_2)

    X_hat = tensor_refold(X_2_hat, X_2_recovery_map, data.shape) + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*i

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    nfe_mrlr_invres_1.append(100*error/norm)
    params_mrlr_invres_1.append(n_elements)

# MRLR-RES-INV-2
partition = [[2], [0, 1]]
rank_2 = 4
nfe_mrlr_invres_2 = []
params_mrlr_invres_2 = []
for i in ranks:
    X_3_hat = parafac_decomposition(data, i)
    R_3, R_3_recovery_map = tensor_unfold(X_3_hat, partition)

    X_2, X_2_recovery_map = tensor_unfold(data, partition)
    X_2_hat = parafac_decomposition(X_2 - R_3, rank_2)

    X_hat = tensor_refold(X_2_hat, X_2_recovery_map, data.shape) + X_3_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2\
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*i

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    nfe_mrlr_invres_2.append(100*error/norm)
    params_mrlr_invres_2.append(n_elements)

# PLOT
pparam = dict(ylabel='$NFE (\%)$', xlabel='$\# Parameters$')
with plt.style.context(['science'], ['ieee']):
    fig, ax = plt.subplots()
    plt.grid()
    ax.plot(params_parafac, nfe_parafac, label="PARAFAC", marker='o', c='k')
    ax.plot(params_mrlr_res_1, nfe_mrlr_res_1, label="MRLR-res-1", marker='D', c='orange')
    ax.plot(params_mrlr_res_2, nfe_mrlr_res_2, label="MRLR-res-2", marker='x', c='blue')
    ax.plot(params_mrlr_invres_1, nfe_mrlr_invres_1, label="MRLR-res-1-reverse", marker='s', c='r')
    ax.plot(params_mrlr_invres_2, nfe_mrlr_invres_2, label="MRLR-res-2-reverse", marker='p', c='g')
    ax.legend()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0., 1.5)
    ax.set_xlim(1500., 30000)
    fig.savefig('results/fig_1.jpg', dpi=300)
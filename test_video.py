import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorly as tl
from tensorly.tenalg import khatri_rao as kr
from tensorly.cp_tensor import unfolding_dot_khatri_rao
from tensorly.decomposition import parafac

from utils import reindex, tensor_unfold, tensor_refold

os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

video_raw = cv2.VideoCapture("data/sunset.mp4")
n_frames = int(video_raw.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video_raw.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_raw.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale_percent = 5 # percent of original size
width = int(width * scale_percent / 100)
height = int(height * scale_percent / 100)
dim = (width, height)

data = np.empty((n_frames, height, width, 3), np.dtype('uint8'))
for i in range(n_frames):
    ret, frame = video_raw.read()
    frame = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    data[i] = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    if not ret:
        print(i)
        break
data = data[np.arange(0, 170, 20)]
video_raw.release()

# PARAFAC
frob_cpd = []
param_cpd = []
norm = np.linalg.norm(data.ravel())
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    weights, factors = parafac(
        tl.tensor(data, dtype=tl.float32),
        rank=i,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    full_tensor = tl.kruskal_to_tensor((weights, factors))
    error = np.linalg.norm(data.ravel() - full_tensor.ravel())
    frob_cpd.append(100*error/norm)
    param_cpd.append(np.sum(data.shape)*i)

# MRLR
frob_mrlr = []
param_mrlr = []
norm = np.linalg.norm(data.ravel())
for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    rank = i
    rank_2 = 1
    rank_3 = 1

    X_2, X_2_recovery_map = tensor_unfold(data, [[0, 1], [2, 3]])
    X_3, X_3_recovery_map = tensor_unfold(data, [[0], [1],[2, 3]])
    X_4 = data

    X_target_2 = X_2
    weights, factors_2_hat = parafac(
        tl.tensor(X_target_2, dtype=tl.float32),
        rank=rank_2,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_2_hat = tl.kruskal_to_tensor((weights, factors_2_hat))
    X_2_rec = tensor_refold(X_2_hat, X_2_recovery_map, data.shape)

    R_2, R_2_recovery_map = tensor_unfold(X_2_rec, [[0], [1],[2, 3]])
    X_target_3 = X_3 - R_2
    weights, factors_3_hat = parafac(
        tl.tensor(X_target_3, dtype=tl.float32),
        rank=rank_3,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_3_hat = tl.kruskal_to_tensor((weights, factors_3_hat))
    X_3_rec = tensor_refold(X_3_hat, X_3_recovery_map, data.shape)

    X_target_4 = X_4 - X_2_rec - X_3_rec
    weights, factors_4_hat = parafac(
        tl.tensor(X_target_4, dtype=tl.float32),
        rank=rank,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_4_hat = tl.kruskal_to_tensor((weights, factors_4_hat))

    X_hat = X_2_rec + X_3_rec + X_4_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2 \
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank_3 \
        + (X_4_hat.shape[0] + X_4_hat.shape[1] + X_4_hat.shape[2] + X_4_hat.shape[3])*rank

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    frob_mrlr.append(100*error/norm)
    param_mrlr.append(n_elements)

# MRLR-INVERSE
frob_mrlr_reverse = []
param_mrlr_reverse = []
norm = np.linalg.norm(data.ravel())
for i in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    rank = i
    rank_2 = 1
    rank_3 = 1

    X_2, X_2_recovery_map = tensor_unfold(data, [[0, 1], [2, 3]])
    X_3, X_3_recovery_map = tensor_unfold(data, [[0], [1],[2, 3]])
    X_4 = data

    X_target_4 = data
    weights, factors_4_hat = parafac(
        tl.tensor(X_target_4, dtype=tl.float32),
        rank=rank,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_4_hat = tl.kruskal_to_tensor((weights, factors_4_hat))

    R_4, _ = tensor_unfold(X_4_hat, [[0], [1],[2, 3]])
    X_target_3 = X_3 - R_4
    weights, factors_3_hat = parafac(
        tl.tensor(X_target_3, dtype=tl.float32),
        rank=rank_3,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_3_hat = tl.kruskal_to_tensor((weights, factors_3_hat))
    X_3_rec = tensor_refold(X_3_hat, X_3_recovery_map, data.shape)

    R_4, _ = tensor_unfold(X_4_hat, [[0, 1],[2, 3]])
    R_3, _ = tensor_unfold(X_3_rec, [[0, 1],[2, 3]])
    X_target_2 = X_2 - R_4 - R_3
    weights, factors_2_hat = parafac(
        tl.tensor(X_target_2, dtype=tl.float32),
        rank=rank_2,
        normalize_factors=False,
        init='random',
        tol=1e-3
    )
    X_2_hat = tl.kruskal_to_tensor((weights, factors_2_hat))
    X_2_rec = tensor_refold(X_2_hat, X_2_recovery_map, data.shape)

    X_hat = X_2_rec + X_3_rec + X_4_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2 \
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank_3 \
        + (X_4_hat.shape[0] + X_4_hat.shape[1] + X_4_hat.shape[2] + X_4_hat.shape[3])*rank

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    frob_mrlr_reverse.append(100*error/norm)
    param_mrlr_reverse.append(n_elements)

# PLOT
pparam = dict(ylabel='$NFE (\%)$', xlabel='$\# Parameters$')
with plt.style.context(['science'], ['ieee']):
    fig, ax = plt.subplots()
    plt.grid()
    ax.plot(param_cpd, frob_cpd, label="PARAFAC", marker='o', c='k')
    ax.plot(param_mrlr, frob_mrlr, label="MRLR", marker='D', c='orange')
    ax.plot(param_mrlr_reverse, frob_mrlr_reverse, label="MRLR-reverse", marker='x', c='blue')
    ax.legend()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0, 10)
    fig.savefig('results/fig_2.jpg', dpi=300)
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import reindex, tensor_unfold, tensor_refold, parafac_decomposition

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

ranks = [1] + list(range(10, 110, 10))
partition_2 = [[0, 1], [2, 3]]
partition_3 = [[0], [1],[2, 3]]
rank_2 = 1
rank_3 = 1
norm = np.linalg.norm(data.ravel())

# PARAFAC
nfe_parafac = list()
params_parafac = list()
for i in ranks:
    tensor_hat = parafac_decomposition(data, i)
    error = np.linalg.norm(data.ravel() - tensor_hat.ravel())
    nfe_parafac.append(100*error/norm)
    params_parafac.append(np.sum(data.shape)*i)

# MRLR
nfe_mrlr = []
params_mrlr = []
for i in ranks:
    X_2, X_2_recovery_map = tensor_unfold(data, partition_2)
    X_2_hat = parafac_decomposition(X_2, rank_2)
    X_2_rec = tensor_refold(X_2_hat, X_2_recovery_map, data.shape)
    R_2, R_2_recovery_map = tensor_unfold(X_2_rec, partition_3)

    X_3, X_3_recovery_map = tensor_unfold(data, partition_3)
    X_3_hat = parafac_decomposition(X_3 - R_2, rank_3)
    X_3_rec = tensor_refold(X_3_hat, X_3_recovery_map, data.shape)

    X_4_hat = parafac_decomposition(data - X_2_rec - X_3_rec, i)

    X_hat = X_2_rec + X_3_rec + X_4_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2 \
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank_3 \
        + (X_4_hat.shape[0] + X_4_hat.shape[1] + X_4_hat.shape[2] + X_4_hat.shape[3])*i

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    nfe_mrlr.append(100*error/norm)
    params_mrlr.append(n_elements)

# MRLR-INVERSE
nfe_mrlr_reverse = []
params_mrlr_reverse = []
for i in ranks:
    X_4_hat = parafac_decomposition(data, i)
    R_4, _ = tensor_unfold(X_4_hat, partition_3)

    X_3, X_3_recovery_map = tensor_unfold(data, partition_3)
    X_3_hat = parafac_decomposition(X_3 - R_4, rank_3)
    X_3_rec = tensor_refold(X_3_hat, X_3_recovery_map, data.shape)
    R_4, _ = tensor_unfold(X_4_hat, partition_2)
    R_3, _ = tensor_unfold(X_3_rec, partition_2)

    X_2_hat = parafac_decomposition(X_2 - R_4 - R_3, rank_2)
    X_2_rec = tensor_refold(X_2_hat, X_2_recovery_map, data.shape)

    X_hat = X_2_rec + X_3_rec + X_4_hat
    n_elements = (X_2_hat.shape[0] + X_2_hat.shape[1])*rank_2 \
        + (X_3_hat.shape[0] + X_3_hat.shape[1] + X_3_hat.shape[2])*rank_3 \
        + (X_4_hat.shape[0] + X_4_hat.shape[1] + X_4_hat.shape[2] + X_4_hat.shape[3])*i

    error = np.linalg.norm(data.ravel() - X_hat.ravel())
    nfe_mrlr_reverse.append(100*error/norm)
    params_mrlr_reverse.append(n_elements)

# PLOT
pparam = dict(ylabel='$NFE (\%)$', xlabel='$\# Parameters$')
with plt.style.context(['science'], ['ieee']):
    fig, ax = plt.subplots()
    plt.grid()
    ax.plot(params_parafac, nfe_parafac, label="PARAFAC", marker='o', c='k')
    ax.plot(params_mrlr, nfe_mrlr, label="MRLR", marker='D', c='orange')
    ax.plot(params_mrlr_reverse, nfe_mrlr_reverse, label="MRLR-reverse", marker='x', c='blue')
    ax.legend()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    ax.set_ylim(0, 10)
    fig.savefig('results/fig_2.jpg', dpi=300)
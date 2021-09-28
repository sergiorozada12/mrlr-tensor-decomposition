import itertools
import numpy as np

import tensorly as tl
from tensorly.decomposition import parafac

def parafac_decomposition(data, rank):
    weights, factors = parafac(
        tl.tensor(data, dtype=tl.float32),
        rank=rank,
        normalize_factors=False,
        init='random',
        tol=1e-3,
        n_iter_max=100
    )
    return tl.kruskal_to_tensor((weights, factors))

def reindex(tensor, partitions, index):
    new_index = [0]*len(partitions)
    for i in range(len(partitions)):
        partition = partitions[i]
        dimension_1 = partition[0]
        n_p_1 = index[dimension_1]
        if len(partition) == 1:
            new_index[i] = n_p_1
            continue
        result = n_p_1
        for j in range(1, len(partition)):
            dimension = partition[j]
            n_p_k = index[dimension]
            prod = np.prod([tensor.shape[partition[k]] for k in range(j)])
            result += n_p_k*prod
        new_index[i] = int(result)
    return tuple(new_index)

def tensor_unfold(tensor, partitions):
    generators = [list(range(dimension_shape)) for dimension_shape in tensor.shape]
    indices = itertools.product(*generators)
    
    new_shape = [0]*len(partitions)
    for idx, partition in enumerate(partitions):
        new_shape[idx] = np.prod([tensor.shape[dimension] for dimension in partition])
    
    idx_recovery_map = {}
    new_tensor = np.zeros(new_shape)
    for idx in indices:
        new_idx = reindex(tensor, partitions, idx)
        new_tensor[new_idx] = tensor[idx]
        idx_recovery_map[new_idx] = idx
    return new_tensor, idx_recovery_map

def tensor_refold(tensor, recovery_map, new_shape):
    tensor_recovered = np.zeros(new_shape)
    for idx in recovery_map:
        tensor_recovered[recovery_map[idx]] = tensor[idx]
    return tensor_recovered
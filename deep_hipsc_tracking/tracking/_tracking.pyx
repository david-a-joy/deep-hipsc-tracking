# cython: language_level=3

import numpy as np
cimport numpy as np

from sklearn.neighbors import BallTree

BOOL_TYPE = np.uint8
INT_TYPE = np.int
FLOAT_TYPE = np.float32

ctypedef np.uint32_t INT_TYPE_t
ctypedef np.float32_t FLOAT_TYPE_t


cdef _remove_duplicates(np.ndarray[FLOAT_TYPE_t, ndim=1] dist,
                        np.ndarray[INT_TYPE_t, ndim=1] ind,
                        FLOAT_TYPE_t max_dist):
    """ Find the index of the smallest array and return that """

    assert dist.shape[0] == ind.shape[0]

    cdef INT_TYPE_t i, cur_ind, best_dist
    cdef np.ndarray[INT_TYPE_t, ndim=1] idx, unique_ind, final_ind, final_idx
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] final_dist
    cdef np.ndarray mask

    unique_ind = np.unique(ind)
    idx = np.arange(dist.shape[0], dtype=np.uint32)

    final_dist = np.empty_like(unique_ind, dtype=np.float32)
    final_ind = np.empty_like(unique_ind, dtype=np.uint32)
    final_idx = np.empty_like(unique_ind, dtype=np.uint32)

    for i, cur_ind in enumerate(unique_ind):
        mask_ind = ind == cur_ind
        masked_idx = idx[mask_ind]

        best_ind = np.argmin(dist[mask_ind])

        final_dist[i] = dist[mask_ind][best_ind]
        final_ind[i] = cur_ind
        final_idx[i] = masked_idx[best_ind]

    mask = final_dist < max_dist
    return final_ind[mask], final_idx[mask]


def remove_duplicates(np.ndarray dist, np.ndarray ind, FLOAT_TYPE_t max_dist):
    """ Remove the duplication from the array """

    dist = np.squeeze(dist).astype(np.float32)
    ind = np.squeeze(ind).astype(np.uint32)

    if dist.ndim == 0:
        dist = np.array([dist])
    if ind.ndim == 0:
        ind = np.array([ind])

    if dist.ndim != 1:
        raise ValueError('dist must be 1D')
    if ind.ndim != 1:
        raise ValueError('ind must be 1D')
    return _remove_duplicates(dist, ind, max_dist)


def link_chains(int tp2, np.ndarray t2,
                np.ndarray ind1, np.ndarray ind2,
                np.ndarray index, list chains):

    cdef np.ndarray[INT_TYPE_t, ndim=1] new_index, new_locs
    cdef np.ndarray new_mask
    cdef INT_TYPE_t i1, i2, chain_idx, max_index
    cdef list chain

    ind1 = ind1.astype(np.int)
    ind2 = ind2.astype(np.int)
    assert ind1.shape[0] == ind2.shape[0]
    max_index = index.shape[0]

    new_index = np.empty((t2.shape[0], ), dtype=np.uint32)

    # Load all the old points
    new_mask = np.ones((t2.shape[0], ), dtype=np.bool)
    new_mask[ind2] = False

    for i1, i2 in zip(ind1, ind2):
        # Handle the rare case where we make a bad link
        if i1 >= max_index:
            new_mask[i2] = True
        else:
            chain_idx = index[i1]
            chain = chains[int(chain_idx)].append((tp2, t2[i2, 0], t2[i2, 1]))
            new_index[i2] = chain_idx

    # Load all the new points
    new_locs = np.arange(t2.shape[0], dtype=np.uint32)[new_mask]
    for i2 in new_locs:
        chain_idx = len(chains)
        chains.append([(tp2, t2[i2, 0], t2[i2, 1])])
        new_index[i2] = chain_idx

    return new_index, chains

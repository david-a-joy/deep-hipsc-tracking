# cython: language_level=3

import numpy as np
cimport numpy as np

cimport cython

INT_TYPE = np.int32
FLOAT_TYPE = np.float64

ctypedef np.uint32_t INT_TYPE_t
ctypedef np.float64_t FLOAT_TYPE_t

# Constants
ALPHA = 0.3  # Radius for the basin of attraction

# Smoothing parameters
# Low beta, very soft assignment
# High beta, very hard assignment
BETA_INIT = 0.5  # Initial smoothing
BETA_FINAL = 10  # Final smoothing
BETA_RATE = 1.075  # Smoothing rate

# Radius parameters
RADIUS_INIT = 2  # Initial minimum radius
RADIUS_RATE = 0.99  # Decay rate for radius

# Normalization parameters
MAX_NORM_ITERS = 100  # Maximum iterations for Sinkhorn's method
NORM_TOL = 1e-3  # Tolerance for normalization of the matrix

# Functions


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _norm_matrix(np.ndarray[FLOAT_TYPE_t, ndim=2] matrix,
                  int rows,
                  int cols,
                  int max_norm_iters=MAX_NORM_ITERS,
                  float norm_tol=NORM_TOL):
    """ Sinkhorn's method to normalize a matrix """

    cdef unsigned int i
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] sum_row, sum_col
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] matrix_row, matrix_col

    # rows, cols = matrix.shape

    for i in range(max_norm_iters):
        sum_row = np.sum(matrix, axis=1)[:, np.newaxis]
        matrix_row = matrix / sum_row

        sum_col = np.sum(matrix_row, axis=0)[np.newaxis, :]
        matrix_col = matrix_row / sum_col

        if np.all(np.abs(matrix_col - matrix) < norm_tol):
            return matrix_col
        matrix = matrix_col
    return matrix


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _estimate_probs(np.ndarray[FLOAT_TYPE_t, ndim=2] points1,
                     np.ndarray[FLOAT_TYPE_t, ndim=2] points2,
                     np.ndarray[FLOAT_TYPE_t, ndim=2] translation,
                     int n, int m,
                     float alpha=ALPHA,
                     float radius=1.0):
    """ Estimate the pairwise probability for aligning points1/points2

    """

    cdef int i, j
    cdef float min_score, score

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] delta, p1, p2
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] probs

    # Unpack things
    # n = points1.shape[0]
    # assert points1.shape[1] == 2
    # m = points2.shape[0]
    # assert points2.shape[1] == 2

    probs = np.zeros((n+1, m+1))
    min_score = -radius**2

    for i in range(n):
        for j in range(m):
            p1 = points1[i, :]
            p2 = points2[j, :]

            delta = p1 - translation[i, :] - p2
            score = -(delta[0]**2 + delta[1]**2 - alpha)

            # Sparse prior
            if score < min_score:
                score = -np.inf
            probs[i, j] = score
    return probs


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _update_params(np.ndarray[FLOAT_TYPE_t, ndim=2] points1,
                    np.ndarray[FLOAT_TYPE_t, ndim=2] points2,
                    np.ndarray[FLOAT_TYPE_t, ndim=2] matches,
                    int n, int m):
    """ Update the deterministic parameters

    Eqn 3.3
    """

    cdef int i, j
    cdef float min_score, score

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] p1, p2
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] new_trans

    # Unpack things
    # n = points1.shape[0]
    # assert points1.shape[1] == 2
    # m = points2.shape[0]
    # assert points2.shape[1] == 2

    new_trans = np.zeros((n, 2))

    for i in range(n):
        for j in range(m):
            p1 = points1[i, :]
            p2 = points2[j, :]

            new_trans[i, :] += matches[i, j] * (p1 - p2)
    return new_trans  # Don't have to norm because axis sum to 1


def soft_assign(points1, points2,
                float beta_init=BETA_INIT,
                float beta_final=BETA_FINAL,
                float beta_rate=BETA_RATE,
                float radius_init=RADIUS_INIT,
                float radius_rate=RADIUS_RATE,
                int max_norm_iters=MAX_NORM_ITERS):

    cdef int n, m, i
    cdef float beta, radius
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] translation, matches, probs
    cdef list pairs, outliers_i, outliers_j
    cdef set used_j

    # Unpack things
    n = points1.shape[0]
    assert points1.shape[1] == 2
    m = points2.shape[0]
    assert points2.shape[1] == 2

    translation = np.zeros((n, 2))

    beta = beta_init
    radius = radius_init
    while beta < beta_final:
        probs = _estimate_probs(points1, points2, translation,
                                radius=radius, alpha=ALPHA,
                                n=n, m=m)

        matches = np.exp(beta * probs)
        matches = _norm_matrix(matches, rows=n+1, cols=m+1)

        translation = _update_params(
            points1, points2, matches,
            n=n, m=m)

        beta *= beta_rate
        radius *= radius_rate

    pairs = []
    outliers_i = []
    used_j = set()
    for i in range(n):
        bestj = np.argmax(matches[i, :])
        if bestj < m:
            pairs.append((i, bestj))
            used_j.add(bestj)
        else:
            outliers_i.append(i)

    outliers_j = [j for j in range(m) if j not in used_j]
    return np.array(pairs, dtype=np.uint), translation, outliers_i, outliers_j


def demo_softassign():
    points1 = np.random.rand(100, 2)
    points2 = np.random.rand(100, 2)
    return soft_assign(points1, points2, radius_init=1.0)

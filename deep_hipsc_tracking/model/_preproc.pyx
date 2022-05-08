""" Cythonized functions for preprocessing """
# cython: language_level=3

import numpy as np
cimport numpy as np

cimport cython

BOOL_TYPE = np.uint8
INT_TYPE = np.int64
FLOAT_TYPE = np.float32

ctypedef np.int_t INT_TYPE_t
ctypedef np.float32_t FLOAT_TYPE_t
ctypedef np.uint8_t BOOL_TYPE_t

cdef enum:
    MODE_PEAK = 1
    MODE_MEAN = 2
    MODE_CONV = 3


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _collapse_batch_peak(np.ndarray[FLOAT_TYPE_t, ndim=4] batch_slab,
                               np.ndarray[INT_TYPE_t, ndim=2] batch_idx,
                               int batch_len,
                               object detector,
                               np.ndarray[FLOAT_TYPE_t, ndim=2] response,
                               np.ndarray[INT_TYPE_t, ndim=2] counts,
                               int rows,
                               int cols,
                               int srows,
                               int scols,
                               int crows,
                               int ccols,
                               int batch_stride):

    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] batch_resp
    cdef int ist, jst, ied, jed, ict, jct, hw, idx

    batch_resp = detector.predict(batch_slab, batch_size=batch_len)

    for idx in range(batch_len):
        ist, jst = batch_idx[idx, :]

        ied, jed = ist + srows, jst + scols
        ict, jct = (ist+ied)//2, (jst+jed)//2

        hw = (batch_stride - 1)//2
        ist, ied = ict-hw, ict+hw+1
        jst, jed = jct-hw, jct+hw+1

        ist = max([ist, 0])
        ied = min([ied, rows])

        jst = max([jst, 0])
        jed = min([jed, cols])

        response[ist:ied, jst:jed] += batch_resp[idx, :]
        counts[ist:ied, jst:jed] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _collapse_batch_conv(np.ndarray[FLOAT_TYPE_t, ndim=4] batch_slab,
                               np.ndarray[INT_TYPE_t, ndim=2] batch_idx,
                               int batch_len,
                               object detector,
                               np.ndarray[FLOAT_TYPE_t, ndim=2] response,
                               np.ndarray[INT_TYPE_t, ndim=2] counts,
                               int rows,
                               int cols,
                               int srows,
                               int scols,
                               int crows,
                               int ccols,
                               int batch_stride):

    cdef np.ndarray[FLOAT_TYPE_t, ndim=4] batch_resp
    cdef int ist, jst, ied, jed, ict, jct, hw, idx

    batch_resp = detector.predict(batch_slab, batch_size=batch_len).astype(FLOAT_TYPE)

    for idx in range(batch_len):
        ist, jst = batch_idx[idx, :]

        ied, jed = ist + crows, jst + ccols

        ist = max([ist, 0])
        ied = min([ied, rows])

        jst = max([jst, 0])
        jed = min([jed, cols])

        response[ist:ied, jst:jed] += batch_resp[idx, :, :, 0]
        counts[ist:ied, jst:jed] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _collapse_batch_mean(np.ndarray[FLOAT_TYPE_t, ndim=4] batch_slab,
                               np.ndarray[INT_TYPE_t, ndim=2] batch_idx,
                               int batch_len,
                               object detector,
                               np.ndarray[FLOAT_TYPE_t, ndim=2] response,
                               np.ndarray[INT_TYPE_t, ndim=2] counts,
                               int rows,
                               int cols,
                               int srows,
                               int scols,
                               int crows,
                               int ccols,
                               int batch_stride):

    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] batch_resp
    cdef int ist, jst, idx

    batch_resp = detector.predict(batch_slab, batch_size=batch_len)

    for idx in range(batch_len):
        ist, jst = batch_idx[idx, :]
        response[ist:ist+srows, jst:jst+scols] += batch_resp[idx, :]
        counts[ist:ist+srows, jst:jst+scols] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _composite_mask(np.ndarray[dtype=FLOAT_TYPE_t, ndim=3] img,
                     np.ndarray[dtype=BOOL_TYPE_t, ndim=2] mask,
                     object detector,
                     int rows,
                     int cols,
                     int colors,
                     int srows,
                     int scols,
                     int crows,
                     int ccols,
                     int batch_size,
                     int batch_stride,
                     int mode):
    """ Composite the mask """

    cdef int i, j, bidx
    cdef np.ndarray[FLOAT_TYPE_t, ndim=4] batch_slab
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] response
    cdef np.ndarray[INT_TYPE_t, ndim=2] batch_idx, counts
    cdef np.ndarray count_mask

    batch_slab = np.empty((batch_size, srows, scols, colors), dtype=FLOAT_TYPE)
    batch_idx = np.empty((batch_size, 2), dtype=INT_TYPE)
    bidx = 0

    response = np.zeros((rows, cols), dtype=FLOAT_TYPE)
    counts = np.zeros((rows, cols), dtype=INT_TYPE)

    for i in range(0, rows - srows + 1, batch_stride):
        for j in range(0, cols - scols + 1, batch_stride):
            if not np.any(mask[i:i+srows, j:j+scols] == 1):
                continue

            batch_slab[bidx, :, :] = img[i:i+srows, j:j+scols, :]
            batch_idx[bidx, 0] = i
            batch_idx[bidx, 1] = j
            bidx += 1

            if bidx >= batch_size:
                if mode == MODE_PEAK:
                    _collapse_batch_peak(batch_slab=batch_slab,
                                         batch_idx=batch_idx,
                                         batch_len=batch_size,
                                         detector=detector,
                                         response=response,
                                         counts=counts,
                                         rows=rows,
                                         cols=cols,
                                         srows=srows,
                                         scols=scols,
                                         crows=crows,
                                         ccols=ccols,
                                         batch_stride=batch_stride)
                elif mode == MODE_CONV:
                    _collapse_batch_conv(batch_slab=batch_slab,
                                         batch_idx=batch_idx,
                                         batch_len=batch_size,
                                         detector=detector,
                                         response=response,
                                         counts=counts,
                                         rows=rows,
                                         cols=cols,
                                         srows=srows,
                                         scols=scols,
                                         crows=crows,
                                         ccols=ccols,
                                         batch_stride=batch_stride)
                elif mode == MODE_MEAN:
                    _collapse_batch_mean(batch_slab=batch_slab,
                                         batch_idx=batch_idx,
                                         batch_len=batch_size,
                                         detector=detector,
                                         response=response,
                                         counts=counts,
                                         rows=rows,
                                         cols=cols,
                                         srows=srows,
                                         scols=scols,
                                         crows=crows,
                                         ccols=ccols,
                                         batch_stride=batch_stride)
                else:
                    assert False, "Invalid mode"
                bidx = 0

    if bidx > 0:
        if mode == MODE_PEAK:
            _collapse_batch_peak(batch_slab=batch_slab[:bidx, :, :],
                                 batch_idx=batch_idx[:bidx, :],
                                 batch_len=bidx,
                                 detector=detector,
                                 response=response,
                                 counts=counts,
                                 rows=rows,
                                 cols=cols,
                                 srows=srows,
                                 scols=scols,
                                 crows=crows,
                                 ccols=ccols,
                                 batch_stride=batch_stride)
        elif mode == MODE_CONV:
            _collapse_batch_conv(batch_slab=batch_slab,
                                 batch_idx=batch_idx,
                                 batch_len=batch_size,
                                 detector=detector,
                                 response=response,
                                 counts=counts,
                                 rows=rows,
                                 cols=cols,
                                 srows=srows,
                                 scols=scols,
                                 crows=crows,
                                 ccols=ccols,
                                 batch_stride=batch_stride)
        elif mode == MODE_MEAN:
            _collapse_batch_mean(batch_slab=batch_slab[:bidx, :, :],
                                 batch_idx=batch_idx[:bidx, :],
                                 batch_len=bidx,
                                 detector=detector,
                                 response=response,
                                 counts=counts,
                                 rows=rows,
                                 cols=cols,
                                 srows=srows,
                                 scols=scols,
                                 crows=crows,
                                 ccols=ccols,
                                 batch_stride=batch_stride)
        else:
            assert False, "Invalid mode"

    count_mask = np.logical_and(counts > 0, mask)
    response[count_mask] = response[count_mask] / counts[count_mask]
    response[~count_mask] = np.nan
    return response


def composite_mask(img, detector, srows, scols,
                   batch_size=2048, batch_stride=5,
                   mode='peak', mask=None, transforms=None,
                   crows=0, ccols=0):
    """ Composite the mask

    :param img:
        The numpy array for the image
    :param detector:
        The machine learning object
    :param srows:
        The row size of the window to sample in
    :param scols:
        The column size of the window to sample in
    :param batch_size:
        How many windows to sample
    :param batch_stride:
        Spacing between centers of the window
    :param mode:
        The mode to composite the samples (mean, peak)
    :param mask:
        If not None, the mask to composite with
    :param transforms:
        If not None, "rotations" tries all four rotations of the image and averages them
    :returns:
        The composited image, the same size as img
    """
    print('Sampling detections...')

    # Pad the image to be square in the size of the sample window
    rows, cols, colors = img.shape
    row_padding = rows % srows
    col_padding = cols % scols

    row_left = row_padding // 2
    col_left = col_padding // 2

    row_right = row_left + rows
    col_right = col_left + cols

    pad_img = np.zeros((rows + row_padding, cols + col_padding, colors),
                       dtype=FLOAT_TYPE)
    pad_img[row_left:row_right, col_left:col_right, :] = img.astype(FLOAT_TYPE)

    if mask is None:
        pad_mask = np.ones((rows + row_padding, cols + col_padding),
                           dtype=BOOL_TYPE)
    else:
        pad_mask = np.zeros((rows + row_padding, cols + col_padding),
                            dtype=BOOL_TYPE)
        pad_mask[row_left:row_right, col_left:col_right] = np.squeeze(mask.astype(BOOL_TYPE))

    if mode in ('peak', 'peaks'):
        mode = MODE_PEAK
    elif mode == 'mean':
        mode = MODE_MEAN
    elif mode == 'conv':
        mode = MODE_CONV
    else:
        raise ValueError('Unknown mode "{}"'.format(mode))

    if transforms in ('none', None):
        rotations = (0, )
    elif transforms == 'rotations':
        rotations = (0, 1, 2, 3)
    else:
        raise ValueError('Unknown transforms: "{}"')

    # Composite the image and rotations of the image
    all_responses = []
    for rotation in rotations:
        if rotation % 4 == 0:
            rimg = pad_img.copy()
            rmask = pad_mask.copy()
        else:
            rimg = np.rot90(pad_img, rotation)
            rmask = np.rot90(pad_mask, rotation)

        rows, cols = rmask.shape
        resp = _composite_mask(img=rimg,
                               mask=rmask,
                               detector=detector,
                               rows=rows,
                               cols=cols,
                               colors=colors,
                               srows=srows,
                               scols=scols,
                               crows=crows,
                               ccols=ccols,
                               batch_size=batch_size,
                               batch_stride=batch_stride,
                               mode=mode)
        if rotation % 4 != 0:
            resp = np.rot90(resp, -rotation)
        all_responses.append(resp[row_left:row_right, col_left:col_right])
    return all_responses

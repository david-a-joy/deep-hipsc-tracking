# cython: language_level=3

# Imports
import numpy as np

cimport numpy as np

cimport cython

INT_TYPE = np.int
FLOAT_TYPE = np.float64

ctypedef np.uint32_t INT_TYPE_t
ctypedef np.float64_t FLOAT_TYPE_t

cdef extern from "math.h":
    double sqrt(double a)


# Functions


@cython.boundscheck(False)
@cython.wraparound(False)
def correlate_arrays(np.ndarray[FLOAT_TYPE_t, ndim=2] final_composite,
                     np.ndarray[INT_TYPE_t, ndim=2] final_counts,
                     np.ndarray[FLOAT_TYPE_t, ndim=2] final_img,
                     int search_step, int blend_x, int blend_y,
                     int xst, int xed, int yst, int yed,
                     int rows, int cols, int comp_rows, int comp_cols,
                     int min_count = 5):
    """ Correlate the overlapping regions between two arrays

    :param ndarray final_composite:
        The final composite array
    :param ndarray final_counts:
        The final counts array

    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] center_composite, center_image
    cdef float best_cc, cc, norm1, norm2, val1, val2, comp_max, image_max, scale
    cdef int xst1, xed1, yst1, yed1, xst2, xed2, yst2, yed2
    cdef int xoff, yoff, shift, y1, y2, x1, x2, count
    cdef int best_xoff, best_yoff
    cdef int best_xst1, best_xst2, best_xed1, best_xed2
    cdef int best_yst1, best_yst2, best_yed1, best_yed2

    # Force the images to be 0-mean, or cosine distance works not great
    center_image = final_img - np.mean(final_img)
    center_composite = final_composite - np.mean(final_composite[final_counts > 0])

    # Initialize the correlation value to a flag value
    best_cc = -2.0
    best_xoff = 0
    best_yoff = 0

    for xoff in range(-search_step, search_step+1):
        for yoff in range(-search_step, search_step+1):
            # Find the current offset into the composite image
            xst1 = xst + xoff
            xed1 = xed + xoff
            yst1 = yst + yoff
            yed1 = yed + yoff

            # Find the current offset into the new/added image
            xst2 = 0
            xed2 = cols
            yst2 = 0
            yed2 = rows

            cc = 0
            count = 0
            norm1 = 0
            norm2 = 0

            # Either do a full comparison
            if blend_x < 0 and blend_y < 0:
                for y2 in range(yst2, yed2):
                    # y-bounds check
                    if y2 < 0 or y2 >= rows:
                        continue
                    y1 = y2 - yst2 + yst1
                    if y1 < 0 or y1 >= comp_rows:
                        continue
                    for x2 in range(xst2, xed2):
                        # x bounds check
                        if x2 < 0 or x2 >= cols:
                            continue
                        x1 = x2 - xst2 + xst1
                        if x1 < 0 or x1 >= comp_cols:
                            continue

                        if final_counts[y1, x1] < 1:
                            continue
                        # Cosine distance
                        val1 = center_composite[y1, x1]
                        val2 = center_image[y2, x2]
                        cc += val1*val2
                        norm1 += val1*val1
                        norm2 += val2*val2
                        count += 1
            else:
                # We only insert along the top left, so skip anything that we couldn't possibly see
                # Top row, all the columns
                for y2 in range(yst2, yst2+blend_y):
                    # y-bounds check
                    if y2 < 0 or y2 >= rows:
                        continue
                    y1 = y2 - yst2 + yst1
                    if y1 < 0 or y1 >= comp_rows:
                        continue
                    for x2 in range(xst2, xed2):
                        # x bounds check
                        if x2 < 0 or x2 >= cols:
                            continue
                        x1 = x2 - xst2 + xst1
                        if x1 < 0 or x1 >= comp_cols:
                            continue

                        if final_counts[y1, x1] < 1:
                            continue
                        # Cosine distance
                        val1 = center_composite[y1, x1]
                        val2 = center_image[y2, x2]
                        cc += val1*val2
                        norm1 += val1*val1
                        norm2 += val2*val2
                        count += 1
                # Top columns, all the rows
                for y2 in range(yst2, yed2):
                    # y-bounds check
                    if y2 < 0 or y2 >= rows:
                        continue
                    y1 = y2 - yst2 + yst1
                    if y1 < 0 or y1 >= comp_rows:
                        continue
                    for x2 in range(xst2, xst2+blend_x):
                        # x bounds check
                        if x2 < 0 or x2 >= cols:
                            continue
                        x1 = x2 - xst2 + xst1
                        if x1 < 0 or x1 >= comp_cols:
                            continue

                        if final_counts[y1, x1] < 1:
                            continue
                        # Cosine distance
                        val1 = center_composite[y1, x1]
                        val2 = center_image[y2, x2]
                        cc += val1*val2
                        norm1 += val1*val1
                        norm2 += val2*val2
                        count += 1

            # If we never got any counts, return
            if count < min_count:
                continue

            # Work out the current correlation for this posititon
            norm_factor = sqrt(norm1) * sqrt(norm2)
            if norm_factor < 1e-5:
                continue
            cc = cc / norm_factor
            if cc > best_cc:
                best_cc = cc
                best_xoff = xoff
                best_yoff = yoff

    # Cool, now work out the boundaries
    if best_cc < -1.0:
        # We never got an offset for anything
        return (-1, -1, -1, -1), (-1, -1, -1, -1)

    # Composite coordinates
    best_yst1 = yst + best_yoff
    best_yed1 = yed + best_yoff
    best_xst1 = xst + best_xoff
    best_xed1 = xed + best_xoff

    # Image coordinates
    best_xst2 = 0
    best_xed2 = cols
    best_yst2 = 0
    best_yed2 = rows

    # If we shifted off the composite image, crop the input image to match
    if best_yst1 < 0:
        best_yst2 = -best_yst1
        best_yst1 = 0
    if best_xst1 < 0:
        best_xst2 = -best_xst1
        best_xst1 = 0
    if best_yed1 > comp_rows:
        best_yed2 = rows + comp_rows - best_yed1
        best_yed1 = comp_rows
    if best_xed1 > comp_cols:
        best_xed2 = cols + comp_cols - best_xed1
        best_xed1 = comp_cols

    return (best_yst1, best_yed1, best_yst2, best_yed2), (best_xst1, best_xed1, best_xst2, best_xed2)


def rolling_slope(np.ndarray x, np.ndarray xp, np.ndarray yp, int window, int order=1):
    """ Find the slope using a rolling window fit

    :param ndarray x:
        x values to evaluate the interpolation at
    :param ndarray xp:
        x samples to interpolate over
    :param ndarray yp:
        y samples to interpolate over
    :param int window:
        window size for the interpolation
    :returns:
        a - the interpolated slopes
    """

    cdef np.ndarray ip, a, mask
    cdef int i, ict0, ict1

    # Sort the xp and yp coordinates
    ip = np.argsort(xp)
    xp = xp[ip]
    yp = yp[ip]

    a = np.zeros_like(x)
    # Handle the left side
    mask = x < xp[window//2]
    a[mask] = np.polyfit(xp[:window], yp[:window], order)[0]

    # Handle the right side
    mask = x >= xp[-window//2]
    a[mask] = np.polyfit(xp[-window:], yp[-window:], order)[0]

    for i in range(xp.shape[0]-window):
        ict0 = i + window//2
        ict1 = ict0 + 1

        mask = np.logical_and(x >= xp[ict0], x < xp[ict1])
        a[mask] = np.polyfit(xp[i:i+window], yp[i:i+window], order)[0]
    return a


def rolling_interp(np.ndarray x, np.ndarray xp, np.ndarray yp, int window, int order=1):
    """ Do a rolling window least squares fit

    :param ndarray x:
        x values to evaluate the interpolation at
    :param ndarray xp:
        x samples to interpolate over
    :param ndarray yp:
        y samples to interpolate over
    :param int window:
        window size for the interpolation
    :returns:
        y - the interpolated y values
    """

    cdef np.ndarray ip, y, mask
    cdef int i, ict0, ict1

    # Sort the xp and yp coordinates
    ip = np.argsort(xp)
    xp = xp[ip]
    yp = yp[ip]

    y = np.zeros_like(x)
    # Handle the left side
    mask = x < xp[window//2]
    y[mask] = np.polyval(np.polyfit(xp[:window], yp[:window], order), x[mask])

    # Handle the right side
    mask = x >= xp[-window//2]
    y[mask] = np.polyval(np.polyfit(xp[-window:], yp[-window:], order), x[mask])

    for i in range(xp.shape[0]-window):
        ict0 = i + window//2
        ict1 = ict0 + 1

        mask = np.logical_and(x >= xp[ict0], x < xp[ict1])
        y[mask] = np.polyval(np.polyfit(xp[i:i+window], yp[i:i+window], order), x[mask])
    return y


# Profiling functions


def profile_correlate_arrays():
    """ Profile the array correlation tool """

    rows, cols = 512, 512
    comp_rows, comp_cols = 1024, 1024

    array1 = np.diag([1.0] * comp_rows)
    array2 = np.diag([1.0] * rows)
    counts = np.zeros((comp_rows, comp_cols), dtype=np.uint32)
    counts[:rows, :] = 1
    counts[:, :cols] = 1

    res = correlate_arrays(final_composite=array1,
                           final_counts=counts,
                           final_img=array2,
                           search_step=10, blend_x=20, blend_y=20,
                           xst=20, xed=20+cols, yst=20, yed=20+rows,
                           rows=rows, cols=cols, comp_rows=comp_rows, comp_cols=comp_cols)

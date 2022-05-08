""" Tracking Utilities

Classes:

* :py:class:`Link`: The main track object to store timeseries of cells

Functions:

* :py:func:`find_flat_regions`: Smooth tracks to find persistent regions
* :py:func:`position_to_velocity`: Convert a track of coordinates to velocities
* :py:func:`smooth_velocity`: Smooth the velocity field
* :py:func:`rolling_window`: Create strided windows over numpy arrays

I/O Functions:

* :py:func:`load_simulation_tsvfile`: Load tracks from Demarcus's simulations
* :py:func:`load_track_csvfile`: Load tracks from the tracking algorithm
* :py:func:`save_track_csvfile`: Save tracks for the tracking algorithm

API Documentation
-----------------

"""

# Imports
import pathlib
from typing import Tuple, List, Optional

# 3rd party
import numpy as np

from scipy.ndimage import label
from scipy.interpolate import UnivariateSpline, interp1d

from skimage.morphology import remove_small_holes, remove_small_objects

# Our own imports
from . import rolling_interp, rolling_slope
from ..utils import BBox

# Classes


class Link(object):
    """ Link object in a chain

    :param \\*args:
        The initial link coordinates (index, tp, x, y)
    """

    def __init__(self, *args):
        if len(args) == 0:
            self.index = None
            self.line_t = []
            self.line_x = []
            self.line_y = []
        else:
            index, tp, x, y = args
            self.index = index
            self.line_t = [tp]
            self.line_x = [x]
            self.line_y = [y]

    @property
    def first(self):
        return self.line_t[0], self.line_x[0], self.line_y[0]

    @property
    def last(self):
        return self.line_t[-1], self.line_x[-1], self.line_y[-1]

    @classmethod
    def merge_mean(cls, *args):
        """ Take the average over several tracks """
        if not all([len(t) == len(args[0]) for t in args]):
            raise ValueError('All tracks must have the same length')

        merge_tt = None
        merge_xx = []
        merge_yy = []
        for track in args:
            tt, xx, yy = track.to_arrays()
            if merge_tt is None:
                merge_tt = tt
            else:
                if not np.allclose(tt, merge_tt):
                    raise ValueError('Time vectors must be aligned')
            merge_xx.append(xx)
            merge_yy.append(yy)
        return cls.from_arrays(tt, np.mean(merge_xx, axis=0), np.mean(merge_yy, axis=0))

    @classmethod
    def join(cls, *args, **kwargs):
        """ Merge two or more links into a single link """
        interp = kwargs.get('interp', None)

        link = cls()
        for a in args:
            if len(link.line_t) >= 2 and interp == 'linear':
                tstep = link.line_t[-1] - link.line_t[-2]
                t0 = link.line_t[-1]
                x0 = link.line_x[-1]
                y0 = link.line_y[-1]
                dt = a.line_t[0] - t0
                dx = a.line_x[0] - x0
                dy = a.line_y[0] - y0

                num_steps = int(round(dt / tstep))

                for i in range(1, num_steps):
                    link.line_t.append(i * dt / num_steps + t0)
                    link.line_x.append(i * dx / num_steps + x0)
                    link.line_y.append(i * dy / num_steps + y0)

            link.line_t += a.line_t
            link.line_x += a.line_x
            link.line_y += a.line_y

        return link

    @classmethod
    def from_tuples(cls, tuples):
        """ Create a link from a list of tuples """

        chain = cls()
        for t, x, y in tuples:
            chain.add(-1, int(t), float(x), float(y))
        return chain

    @classmethod
    def from_arrays(cls, tt, xx, yy):
        """ Create a link from a set of arrays

        :param ndarray tt:
            The time numpy array
        :param ndarry xx:
            The x coordinate array
        :param ndarray yy:
            The y coordinate array
        :returns:
            The link class from the numpy arrays
        """

        assert tt.shape[0] == xx.shape[0]
        assert tt.shape[0] == yy.shape[0]

        chain = cls()
        chain.line_t = list(tt)
        chain.line_x = list(xx)
        chain.line_y = list(yy)
        return chain

    def get_track_length(self, space_scale: float = 1.0) -> float:
        """ Get the arc length along the track

        :returns:
            A float giving the total distance along the track
        """
        _, x, y = self.to_arrays()
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        return np.sum(np.sqrt(dx**2 + dy**2)) * space_scale

    def get_track_displacement(self, space_scale: float = 1.0) -> float:
        """ Get the distance from track start to track end

        :returns:
            A float giving the total displacement of the track
        """
        dx = self.line_x[-1] - self.line_x[0]
        dy = self.line_y[-1] - self.line_y[0]
        return np.sqrt(dx**2 + dy**2) * space_scale

    def get_bbox(self) -> BBox:
        """ Get the bounding box for this track

        :returns:
            The bounding box for this track
        """
        min_x = np.min(self.line_x)
        max_x = np.max(self.line_x)
        min_y = np.min(self.line_y)
        max_y = np.max(self.line_y)
        return BBox(min_x, min_y, max_x, max_y)

    def should_add(self, index, tp, x, y) -> bool:
        """ See if this set of points matches

        :param index:
            The unique key that should match this track
        :param int tp:
            The timepoint for this set
        :param float x:
            The x coordinate for this set
        :param float y:
            The y coordinate for this set
        :returns:
            True if this point fits with the current index
        """
        if self.index is None:
            return True
        dx = self.line_x[-1] - x
        dy = self.line_y[-1] - y
        if index == self.index and np.sqrt(dx**2 + dy**2) < 1e-5:
            return True
        return False

    def add(self, index, tp: int, x: float, y: float):
        """ Add a timepoint

        :param index:
            A unique key for this track
        :param int tp:
            The timepoint for this set
        :param float x:
            The x coordinate for this set
        :param float y:
            The y coordinate for this set
        """
        self.index = index
        self.line_t.append(tp)
        self.line_x.append(x)
        self.line_y.append(y)

    def find_timepoint(self, timepoint: int):
        """ Find the x,y coordinates for a timepoint

        :param int timepoint:
            The timepoint to find
        :returns:
            x, y for the given timepoint or None, None if no match
        """
        try:
            idx = self.line_t.index(timepoint)
        except ValueError:
            return None, None
        else:
            return self.line_x[idx], self.line_y[idx]

    def to_padded_arrays(self,
                         min_t: int = 0,
                         max_t: int = -1,
                         extrapolate: bool = False) -> Tuple[np.ndarray]:
        """ Pad out the arrays in time

        :param int min_t:
            The minimum time index or 0
        :param int max_t:
            The maximum time index or -1 for the last value in our array
        :param bool interpolate:
            If True, interpolate holes in the samples
        :returns:
            The t, x, y numpy arrays, padded with nans
        """
        line_t, line_x, line_y = self.to_arrays()
        if min_t < 0:
            min_t = line_t[min_t]
        if max_t < 0:
            max_t = line_t[max_t]
        pad_t = np.arange(min_t, max_t, 1)

        if extrapolate:
            fill_value_x = (line_x[0], line_x[-1])
            fill_value_y = (line_y[0], line_y[-1])
        else:
            fill_value_x = fill_value_y = np.nan
        fxn_x = interp1d(line_t, line_x,
                         kind='linear',
                         bounds_error=False,
                         fill_value=fill_value_x)
        fxn_y = interp1d(line_t, line_y,
                         kind='linear',
                         bounds_error=False,
                         fill_value=fill_value_y)
        return pad_t, fxn_x(pad_t), fxn_y(pad_t)

    def to_arrays(self) -> Tuple[np.ndarray]:
        """ Convert the internal pointset to numpy arrays

        :returns:
            The t, x, y numpy arrays
        """
        return (np.array(self.line_t),
                np.array(self.line_x),
                np.array(self.line_y))

    def vel_x(self) -> np.ndarray:
        """ X-coordinate velocity

        :returns:
            An (n-1) x 1 array of velocities in the x-axis
        """
        x = np.array(self.line_x)
        t = np.array(self.line_t)
        if x.shape[0] < 2:
            return np.array([0.0])

        dt = t[1:] - t[:-1]
        dx = x[1:] - x[:-1]
        return dx / dt

    def vel_y(self) -> np.ndarray:
        """ Y-coordinate velocity

        :returns:
            An (n-1) x 1 array of velocities in the y-axis
        """
        y = np.array(self.line_y)
        t = np.array(self.line_t)
        if y.shape[0] < 2:
            return np.array([0.0])

        dt = t[1:] - t[:-1]
        dy = y[1:] - y[:-1]
        return dy / dt

    def vel_mag(self) -> np.ndarray:
        """ Velocity magnitude for the point set

        :returns:
            An (n-1) x 1 array of velocity magnitudes
        """
        return np.sqrt(self.vel_x()**2 + self.vel_y()**2)

    def scale_points(self,
                     x_scale: float = 1.0,
                     y_scale: float = 1.0,
                     x_center: Optional[float] = None,
                     y_center: Optional[float] = None):
        """ Scale the track by a factor around its center of mass

        :param float x_scale:
            Scale x by this amount
        :param float y_scale:
            Scale y by this amount
        """
        x = np.array(self.line_x)
        y = np.array(self.line_y)
        if x_center is None:
            x_center = np.mean(x)
        if y_center is None:
            y_center = np.mean(y)

        scaled_x = (x - x_center) * x_scale
        scaled_y = (y - y_center) * y_scale

        self.line_x = list(scaled_x + x_center)
        self.line_y = list(scaled_y + y_center)

    def interpolate_points(self,
                           step: int = 1,
                           smoothing: Optional[int] = None):
        """ Interpolate the points back onto a regular grid

        :param int step:
            The timestep to interpolate over
        :param int smoothing:
            How strongly to smooth the line, s=0 is linear, s=3 is default
        """

        out_t = np.arange(self.line_t[0], self.line_t[-1]+step, step)

        # ext - 3 - return the boundary value
        # Default is ext=0, extrapolate, which does bad things
        x_fit = UnivariateSpline(self.line_t, self.line_x, s=smoothing, ext=3)
        out_x = x_fit(out_t)

        y_fit = UnivariateSpline(self.line_t, self.line_y, s=smoothing, ext=3)
        out_y = y_fit(out_t)

        self.line_t = list(out_t)
        self.line_x = list(out_x)
        self.line_y = list(out_y)

    def __getitem__(self, idx):
        return self.line_t[idx], self.line_x[idx], self.line_y[idx]

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        return all([np.allclose(self.line_t, other.line_t),
                    np.allclose(self.line_x, other.line_x),
                    np.allclose(self.line_y, other.line_y)])

    def __len__(self):
        return len(self.line_x)

    def __repr__(self):
        return f'L({self.line_x},{self.line_y})'


# Functions


def find_flat_regions(tt: np.ndarray,
                      yy: np.ndarray,
                      interp_points: int = 100,
                      cutoff: int = 10,
                      noise_points: int = 5) -> List[np.ndarray]:
    """ Look for regions in of directional persistence or change

    :param ndarray tt:
        The time variable to search over
    :param ndarray yy:
        The space variable to search
    :param int interp_points:
        The number of points to use for the rolling slope fit
    :param float cutoff:
        The largest allowable slope magnitude
    :param int noise_points:
        Number of points to treat a connected region as noise
    :returns:
        A list of masks for each flat region
    """
    slopes = rolling_slope(tt, tt, yy, interp_points)

    # Find all slopes smaller than the cutoff and then convert to individual masks
    mask = (np.abs(slopes) < cutoff).astype(bool)
    if np.any(mask) and not np.all(mask):
        mask = remove_small_objects(mask, noise_points)
        mask = remove_small_holes(mask, noise_points)
        rois = label(mask)[0]
    else:
        rois = mask

    # Split up the connected region labels
    regions = []
    for roi_idx in np.unique(rois):
        if roi_idx < 1:
            continue
        regions.append(rois == roi_idx)
    return regions


def smooth_velocity(tt: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                    resample_factor: int = 10,
                    interp_points: int = 7,
                    smooth_points: int = 5) -> Tuple[np.ndarray]:
    """ Rolling window velocity smoothing

    :param ndarray tt:
        The 1D array of times
    :param ndarray xx:
        The 1D array of x positions
    :param ndarray yy:
        The 1D array of y positions
    :param int resample_factor:
        How many subsamples to take between samples
    :param int interp_points:
        How many support points to use in the linear interpolation
    :param int smooth_points:
        How many support points to use in the rolling average smoothing
    :returns:
        A new set of time, x, y coordinate arrays for the smoothed track
    """

    # Interpolate the points onto a larger space with linear smoothing
    sm_tt = np.linspace(np.min(tt), np.max(tt), resample_factor*tt.shape[0])
    sm_xx = rolling_interp(sm_tt, tt, xx, interp_points, order=1)
    sm_yy = rolling_interp(sm_tt, tt, yy, interp_points, order=1)

    # Remove bumps in the points with rolling averaging
    if smooth_points > 0:
        sm_xx = np.mean(rolling_window(sm_xx, smooth_points*resample_factor, pad='same'), 1)
        sm_yy = np.mean(rolling_window(sm_yy, smooth_points*resample_factor, pad='same'), 1)
    return sm_tt, sm_xx, sm_yy


def position_to_velocity(tt: np.ndarray,
                         xx: np.ndarray,
                         yy: np.ndarray,
                         pad: str = 'valid') -> Tuple[np.ndarray]:
    """ Do the basic position to velocity metric conversion

    :param ndarray tt:
        The 1D array of times
    :param ndarray xx:
        The 1D array of x positions
    :param ndarray yy:
        The 1D array of y positions
    :returns:
        The deltas for time, postion, velocity, and angle
    """
    dt = (tt[1:] - tt[:-1])
    dx = (xx[1:] - xx[:-1])
    dy = (yy[1:] - yy[:-1])
    if pad == 'same':
        dt = np.concatenate([dt, dt[-1]])
        dx = np.concatenate([dx, dx[-1]])
        dy = np.concatenate([dy, dy[-1]])

    ds = np.sqrt((dx**2 + dy**2))
    dv = ds / dt
    dtheta = np.arctan2(dy, dx)
    return dt, dx, dy, ds, dv, dtheta


def load_simulation_tsvfile(track_tsvfile: pathlib.Path) -> List[Link]:
    """ Load the track data from Demarcus' simulations

    :param Path track_tsvfile:
        The tab-separated track data to load
    :returns:
        The list of tracks in the file
    """
    cells = {}

    with track_tsvfile.open('rt') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            line = line.strip()
            if line == '':
                continue
            rec = [r.strip() for r in line.split('\t')]
            if len(rec) != 7:
                raise ValueError(f'Invalid line: {rec}')

            # Unpack the record
            cell_tp = float(rec[0])
            cell_id = int(rec[1])
            cell_x = float(rec[2])
            cell_y = float(rec[3])

            cells.setdefault(cell_id, Link()).add(cell_id, cell_tp, cell_x, cell_y)

    return list(cells.values())


def save_track_csvfile(track_csvfile: pathlib.Path,
                       final_chains: List[Link]):
    """ Save the track CSV file

    :param Path track_csvfile:
        The track file to save
    :param list[Link] final_chains:
        The tracks to save
    """

    track_csvfile.parent.mkdir(exist_ok=True, parents=True)

    with track_csvfile.open('wt') as fp:
        for i, chain in enumerate(final_chains):
            fp.write(f'Track{i:03d}\n')
            chain_t = ','.join(f'{int(t):d}' for t in chain.line_t)
            fp.write(chain_t + '\n')
            chain_x = ','.join(f'{x:0.4f}' for x in chain.line_x)
            fp.write(chain_x + '\n')
            chain_y = ','.join(f'{y:0.4f}' for y in chain.line_y)
            fp.write(chain_y + '\n')


def load_track_csvfile(track_csvfile: pathlib.Path,
                       min_len: int = 0) -> List[Link]:
    """ Load the track csvfile back

    :param Path track_csvfile:
        The file to load
    :param int min_len:
        The minimum length of a track
    :returns:
        A list of Link objects, one for each track
    """

    final_chains = []
    cur_chain_t = None
    cur_chain_x = None
    cur_chain_y = None

    num_tracks = -1

    def append_last_chain():
        if all([c is not None for c in [cur_chain_t, cur_chain_x, cur_chain_y]]):
            assert len(cur_chain_t) == len(cur_chain_x)
            assert len(cur_chain_t) == len(cur_chain_y)

            if min_len <= 0 or len(cur_chain_t) >= min_len:
                chain = Link()
                chain.line_t = list(cur_chain_t)
                chain.line_x = list(cur_chain_x)
                chain.line_y = list(cur_chain_y)
                final_chains.append(chain)

    with track_csvfile.open('rt') as fp:
        for i, line in enumerate(fp):
            line = line.split('#', 1)[0].strip()
            if line == '':
                continue
            if line.startswith('Track'):
                num_tracks += 1
                append_last_chain()
                cur_chain_t = cur_chain_x = cur_chain_y = None
            elif cur_chain_t is None:
                cur_chain_t = [int(t.strip()) for t in line.split(',')]
            elif cur_chain_x is None:
                cur_chain_x = [float(x.strip()) for x in line.split(',')]
            elif cur_chain_y is None:
                cur_chain_y = [float(y.strip()) for y in line.split(',')]
            else:
                raise ValueError(f'Invalid line #{i}: {line}')
    append_last_chain()
    return final_chains


def rolling_window(a, window, pad='valid'):
    """ Rolling window over an array

    :param ndarray a:
        The array to roll a window over
    :param int window:
        The window size
    :returns:
        A window stride object
    """
    orig_shape = a.shape[-1]
    if pad == 'same':
        window_left = window//2
        window_right = window_left + orig_shape
        na = np.empty(a.shape[:-1] + (orig_shape + window, ))
        na[..., window_left:window_right] = a
        na[..., :window_left] = a[..., 0]
        na[..., window_right:] = a[..., -1]
        a = na

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    res = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if pad == 'same':
        res = res[:orig_shape]
    return res

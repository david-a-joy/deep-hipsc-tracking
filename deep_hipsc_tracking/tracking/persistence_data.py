""" Persistence calculation class

Main class:

* :py:class:`PersistenceData`: Calculate and store the persitant velocity traces

Main function:

* :py:func:`calc_track_persistence`: Calculate all the persistence data we want

API Documentation
-----------------

"""

# Imports
import pathlib
from typing import Optional, List, Tuple

# 3rd party
import numpy as np

import matplotlib.pyplot as plt

import h5py

# Our own imports
from . import position_to_velocity, smooth_velocity, find_flat_regions
from ..plotting import set_plot_style

# Constants

PLOT_VELOCITY_MAG_TRACES = False  # Debug plots for seeing the persitance work
PLOT_STYLE = 'light'

MIN_PERSISTENCE_POINTS = 7  # min number of points to calculate persistence
PERSISTENCE_INTERPOLATION = 7  # Interpolate points over this window
PERSISTENCE_SMOOTHING = 5  # Smooth points using this window
PERSISTENCE_DELTA_ANGLE = 2.5  # deg - maximum turning angle to be "persistent"
PERSISTENCE_RESAMPLE_FACTOR = 4  # Resample factor for persistence smoothing


# Classes


class PersistenceData(object):
    """ Store the persistence calculations for a track

    :param ndarray tt:
        The unscaled, uninterpolated time indices
    :param ndarray xx:
        The x coordinates for the track
    :param ndarray yy:
        The y coordinates for the track
    :param float time_scale:
        The conversion from frame number to minutes
    :param int resample_factor:
        How much to subsample and smooth the positions
    """

    def __init__(self, tt: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                 time_scale: Optional[float] = None,
                 space_scale: Optional[float] = None,
                 resample_factor: int = PERSISTENCE_RESAMPLE_FACTOR,
                 interp_points: int = PERSISTENCE_INTERPOLATION,
                 smooth_points: int = PERSISTENCE_SMOOTHING):

        # Parameters
        if time_scale is None:
            time_scale = 1.0
        self.time_scale = time_scale

        if space_scale is None:
            space_scale = 1.0
        self.space_scale = space_scale

        self.resample_factor = resample_factor
        self.interp_points = interp_points
        self.smooth_points = smooth_points

        # Main track data
        self.tt = tt
        self.xx = xx
        self.yy = yy

        # Calculated values
        self.sm_tt, self.sm_xx, self.sm_yy = None, None, None
        self.sm_dt, self.sm_dx, self.sm_dy = None, None, None
        self.sm_ds, self.sm_dv, self.sm_dtheta = None, None, None
        self.sm_unwrap_dtheta = None

        self.pct_quiescent = None
        self.pct_persistent = None

        self.times = None
        self.speeds = None
        self.distances = None
        self.displacements = None
        self.timeline = None
        self.waveform = None
        self.mask = None
        self.gap_times = None

    @classmethod
    def from_hdf5(cls, infile: pathlib.Path):
        """ Load the class from an HDF5 database """

        if isinstance(infile, (str, pathlib.Path)):
            db = h5py.File(str(infile), 'r')
            needs_closing = True
        else:
            db = infile
            needs_closing = False

        try:
            # Load the input args for the class
            tt = np.array(db['tt'])
            xx = np.array(db['xx'])
            yy = np.array(db['yy'])
            time_scale = float(db.attrs['time_scale'])

            obj = cls(tt=tt, xx=xx, yy=yy, time_scale=time_scale)

            # Load the smoothed data
            obj.sm_tt = np.array(db['sm_tt'])
            obj.sm_xx = np.array(db['sm_xx'])
            obj.sm_yy = np.array(db['sm_yy'])

            obj.sm_dt = np.array(db['sm_dt'])
            obj.sm_dx = np.array(db['sm_dx'])
            obj.sm_dy = np.array(db['sm_dy'])

            obj.sm_ds = np.array(db['sm_ds'])
            obj.sm_dv = np.array(db['sm_dv'])
            obj.sm_dtheta = np.array(db['sm_dtheta'])
            obj.sm_unwrap_dtheta = np.array(db['sm_unwrap_dtheta'])

            # Load the persitence calls
            obj.pct_persistent = float(db.attrs['pct_persistent'])
            obj.pct_quiescent = float(db.attrs['pct_quiescent'])

            # Load the per-region properties
            obj.times = [float(t) for t in db['times']]
            obj.gap_times = [float(t) for t in db['gap_times']]

            obj.speeds = [float(s) for s in db['speeds']]
            obj.distances = [float(d) for d in db['distances']]
            obj.displacements = [float(d) for d in db['displacements']]

            # Load whole array properties
            obj.timeline = np.array(db['timeline'])
            obj.waveform = np.array(db['waveform'])
            obj.mask = np.array(db['mask'], dtype=np.bool)

        finally:
            if needs_closing:
                db.close()
        return obj

    def to_hdf5(self, outfile: pathlib.Path):
        """ Save the class to an HDF5 database """

        if isinstance(outfile, (str, pathlib.Path)):
            outfile = pathlib.Path(outfile)
            outfile.parent.mkdir(parents=True, exist_ok=True)

            db = h5py.File(str(outfile), 'w')
            needs_closing = True
        else:
            db = outfile
            needs_closing = False

        try:
            # Save the raw data
            db.create_dataset('tt', data=self.tt)
            db.create_dataset('xx', data=self.xx)
            db.create_dataset('yy', data=self.yy)

            db.attrs['time_scale'] = self.time_scale

            # Save the smoothed data
            db.create_dataset('sm_tt', data=self.sm_tt)
            db.create_dataset('sm_xx', data=self.sm_xx)
            db.create_dataset('sm_yy', data=self.sm_yy)

            db.create_dataset('sm_dt', data=self.sm_dt)
            db.create_dataset('sm_dx', data=self.sm_dx)
            db.create_dataset('sm_dy', data=self.sm_dy)

            db.create_dataset('sm_ds', data=self.sm_ds)
            db.create_dataset('sm_dv', data=self.sm_dv)
            db.create_dataset('sm_dtheta', data=self.sm_dtheta)
            db.create_dataset('sm_unwrap_dtheta', data=self.sm_unwrap_dtheta)

            # Save the persistence calls
            db.attrs['pct_persistent'] = float(self.pct_persistent)
            db.attrs['pct_quiescent'] = float(self.pct_quiescent)

            # Store per-region properties
            db.create_dataset('times', data=np.array(self.times))
            db.create_dataset('gap_times', data=np.array(self.gap_times))

            db.create_dataset('speeds', data=np.array(self.speeds))
            db.create_dataset('distances', data=np.array(self.distances))
            db.create_dataset('displacements', data=np.array(self.displacements))

            # Store whole array properties
            db.create_dataset('timeline', data=self.timeline)
            db.create_dataset('waveform', data=self.waveform)
            db.create_dataset('mask', data=self.mask)

        finally:
            if needs_closing:
                db.close()

    # Helpful attributes

    @property
    def duration(self) -> float:
        """ Total track duration after smoothing """
        return np.sum(self.sm_tt[1:] - self.sm_tt[:-1]) * self.time_scale

    @property
    def distance(self) -> float:
        """ Distance (arc-length) along the entire track """
        dxs = (self.sm_xx[1:] - self.sm_xx[:-1])**2
        dys = (self.sm_yy[1:] - self.sm_yy[:-1])**2
        return np.sum(np.sqrt(dxs + dys)) * self.space_scale

    @property
    def displacement(self) -> float:
        """ Straight line distance between start and end of track """
        dx = (self.sm_xx[-1] - self.sm_xx[0])**2
        dy = (self.sm_yy[-1] - self.sm_yy[0])**2
        return np.sqrt(dx + dy) * self.space_scale

    @property
    def disp_to_dist(self) -> float:
        """ Ratio of displacement to distance: how straight is this track """
        return self.displacement / self.distance

    @property
    def average_velocity(self) -> float:
        """ Average velocity: displacement divided by time """
        return self.displacement / self.duration

    @property
    def average_speed(self) -> float:
        """ Average speed: distance divided by time """
        return self.distance / self.duration

    # Processing Stages

    def resample_positions(self,
                           resample_factor: Optional[int] = None,
                           interp_points: Optional[int] = None,
                           smooth_points: Optional[int] = None):
        """ Smooth positions so we get accurate velocity changes

        :param int resample_factor:
            How much to subsample and smooth the positions
        :param int interp_points:
            Linear window to smooth interpolated positions over
        :param int smooth_points:
            Window to run a mean filter over
        """

        if resample_factor is None:
            resample_factor = self.resample_factor
        if interp_points is None:
            interp_points = self.interp_points
        if smooth_points is None:
            smooth_points = self.smooth_points

        # Still usually need smoothing, even with our current spline fit
        if all([f < 2 for f in [resample_factor, smooth_points, interp_points]]):
            sm_tt, sm_xx, sm_yy = self.tt, self.xx, self.yy
        else:
            sm_tt, sm_xx, sm_yy = smooth_velocity(self.tt, self.xx, self.yy,
                                                  resample_factor=resample_factor,
                                                  interp_points=interp_points,
                                                  smooth_points=smooth_points)

        sm_dt, sm_dx, sm_dy, sm_ds, sm_dv, sm_dtheta = position_to_velocity(
            sm_tt*self.time_scale, sm_xx, sm_yy)

        self.sm_tt, self.sm_xx, self.sm_yy = sm_tt, sm_xx, sm_yy
        self.sm_dt, self.sm_dx, self.sm_dy = sm_dt, sm_dx, sm_dy
        self.sm_ds, self.sm_dv, self.sm_dtheta = sm_ds, sm_dv, sm_dtheta

        self.sm_unwrap_dtheta = np.unwrap(self.sm_dtheta)*180/np.pi

    def find_regions(self,
                     interp_points: Optional[int] = None,
                     cutoff: float = PERSISTENCE_DELTA_ANGLE):
        """ Call all the persistent regions

        :param int interp_points:
            Multiple of the regions to interpolate over
        :param float cutoff:
            Angle to cutoff for persistent vs random motion
        """
        if interp_points is None:
            interp_points = self.interp_points

        rois = find_flat_regions((self.sm_tt)[:-1]*self.time_scale,
                                 self.sm_unwrap_dtheta,
                                 interp_points=interp_points,
                                 cutoff=cutoff)

        # FIXME: This is a stupid place to put this
        if PLOT_VELOCITY_MAG_TRACES:
            dv, dtheta = position_to_velocity(self.tt*self.time_scale, self.xx, self.yy)[-2:]
            raw_signal = (self.tt*self.time_scale, self.xx, self.yy, np.abs(dv), np.unwrap(dtheta)*180/np.pi)
            smooth_signal = (self.sm_tt*self.time_scale, self.sm_xx, self.sm_yy, np.abs(self.sm_dv), self.sm_unwrap_dtheta)
            plot_velocity_mag_dir(raw_signal, smooth_signal, rois=rois)

        # Calculate useful metrics based on the ROIs
        persistence_waveform = np.full_like(self.sm_tt, np.nan)
        persistence_mask = np.zeros_like(self.sm_tt, dtype=np.bool)
        wv_dv = np.append(self.sm_dv, np.array([self.sm_dv[-1]]))

        persistence_times = []
        persistence_speeds = []
        persistence_displacements = []
        persistence_distances = []
        persistence_gap_times = []

        last_end_time = None

        for roi in rois:
            roi_timepoints = (self.sm_tt)[:-1][roi]
            roi_start_time = np.min(roi_timepoints)
            roi_end_time = np.max(roi_timepoints)
            if last_end_time is not None:
                assert roi_start_time > last_end_time
                persistence_gap_times.append(roi_start_time - last_end_time)
            last_end_time = roi_end_time

            roi_mask = np.logical_and(self.sm_tt >= roi_start_time, self.sm_tt < roi_end_time)
            persistence_waveform[roi_mask] = wv_dv[roi_mask]
            persistence_mask[roi_mask] = 1

            roi_time = self.sm_dt[roi]
            roi_dist = self.sm_ds[roi]
            roi_vel = self.sm_dv[roi]
            roi_disp_x = (self.sm_xx[:-1])[roi]
            roi_disp_y = (self.sm_yy[:-1])[roi]

            persistence_times.append(np.sum(roi_time))
            persistence_speeds.append(np.mean(roi_vel))
            persistence_distances.append(np.sum(roi_dist))
            persistence_displacements.append(np.sqrt((roi_disp_y[-1] - roi_disp_y[0])**2 + (roi_disp_x[-1] - roi_disp_x[0])**2))

        # Work out how much this track is persistent vs quiescent
        self.pct_persistent = np.sum(persistence_mask) / persistence_mask.shape[0]
        self.pct_quiescent = 1.0 - self.pct_persistent

        # Store per-timepoint properties
        self.times = persistence_times
        self.gap_times = persistence_gap_times

        self.speeds = persistence_speeds
        self.distances = persistence_distances
        self.displacements = persistence_displacements
        self.timeline = self.sm_tt * self.time_scale
        self.waveform = persistence_waveform
        self.mask = persistence_mask

    def get_timepoint_persistence(self, timepoint: int) -> Tuple[float, float]:
        """ Persistence rate at a given time

        :param int timepoint:
            The timepoint to look at
        :returns:
            The pct_persistent, pct_quiescent at this timepoint
        """
        mask = np.logical_and(self.sm_tt >= timepoint, self.sm_tt < timepoint + 1)
        if np.sum(mask) < 1:
            return np.nan, np.nan
        waveform = self.mask[mask]
        pct_persistent = np.sum(waveform) / np.sum(mask)
        pct_quiescent = 1.0 - pct_persistent
        return pct_persistent, pct_quiescent

    def get_timeline_persistence(self, timeline: np.ndarray) -> np.ndarray:
        """ Convert the track to a timeline

        :param ndarray timeline:
            The timeline to extract waveforms for
        :returns:
            A waveform with samples at each timepoint
        """
        waveform = np.full(timeline.shape, np.nan)
        for idx in range(timeline.shape[0]-1):
            tp0 = timeline[idx]
            tp1 = timeline[idx+1]
            mask = np.logical_and(self.timeline >= tp0, self.timeline < tp1)
            if not mask.any():
                continue
            waveform[idx] = np.nanmean(self.waveform[np.logical_and(mask, self.mask)])
        return waveform


# Helper functions


def plot_velocity_mag_dir(raw_velocity, smooth_velocity,
                          rois: Optional[List[np.ndarray]] = None,
                          plot_style: str = PLOT_STYLE,
                          outfile: Optional[pathlib.Path] = None):
    """ Plot the velocity magnitude and direction data

    :param tuple[Velocity] raw_velocity:
        The raw velocity data
    :param tuple[Velocity] smooth_velocity:
        The smoothed velocity data
    :param list[ndarray] rois:
        A list of boolean masks for interesting parts of the track
    :param str plot_style:
        The style for the plot
    :param Path outfile:
        The file to save to
    """

    if rois is None:
        rois = []

    tt, xx, yy, dv, dtheta = raw_velocity
    sm_tt, sm_xx, sm_yy, sm_dv, sm_dtheta = smooth_velocity

    with set_plot_style(plot_style) as style:
        fig, axes = plt.subplots(1, 3, figsize=(36, 12))
        ax1, ax2, ax3 = axes.ravel()

        ax1.plot(tt[:-1], dtheta, '-r')
        ax1.plot(sm_tt[:-1], sm_dtheta, '-b')
        for mask in rois:
            ax1.plot((sm_tt[:-1])[mask], sm_dtheta[mask], '-b', linewidth=4)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Direction of motion (deg)')

        ax2.plot(tt[:-1], dv, '-r')
        ax2.plot(sm_tt[:-1], sm_dv, '-b')
        for mask in rois:
            ax2.plot((sm_tt[:-1])[mask], sm_dv[mask], '-b', linewidth=4)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Velocity Mag ($\\mu$m/min)')

        ax3.plot(xx, yy, '-r')
        ax3.plot(sm_xx, sm_yy, '-b')
        for mask in rois:
            ax3.plot((sm_xx[:-1])[mask], (sm_yy[:-1])[mask], '-b', linewidth=4)
        ax3.set_xlabel('X Position ($\\mu$m)')
        ax3.set_ylabel('Y Poisiton ($\\mu$m)')

        if outfile is None:
            plt.show()
        else:
            style.savefig(outfile, transparent=True)
            plt.close()


# Functions


def calc_track_persistence(tt: np.ndarray,
                           xx: np.ndarray,
                           yy: np.ndarray,
                           time_scale: Optional[float] = None,
                           space_scale: Optional[float] = None,
                           resample_factor: int = PERSISTENCE_RESAMPLE_FACTOR,
                           interp_points: int = PERSISTENCE_INTERPOLATION,
                           smooth_points: int = PERSISTENCE_SMOOTHING,
                           min_persistence_points: int = MIN_PERSISTENCE_POINTS,
                           cutoff: float = PERSISTENCE_DELTA_ANGLE) -> PersistenceData:
    """ Calculate persistence stats

    :param ndarray tt:
        The time data, **NOT** scaled for the time step
    :param ndarray xx:
        The x-space data, scaled for the space resolution
    :param ndarray yy:
        The y-space data, scaled for the space resolution
    :returns:
        The time smoothed persistence metrics
    """
    # Calculate persistence times for longer tracks
    if tt.shape[0] < min_persistence_points:
        return None

    persistence = PersistenceData(tt=tt, xx=xx, yy=yy,
                                  time_scale=time_scale,
                                  space_scale=space_scale,
                                  resample_factor=resample_factor,
                                  smooth_points=smooth_points,
                                  interp_points=interp_points)
    persistence.resample_positions()
    persistence.find_regions(cutoff=cutoff)
    return persistence

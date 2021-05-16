""" Store the Gridded Track Database

* :py:class:`GridDB`: Main space-time track database

API Documentation
-----------------

"""

# Imports
from typing import Dict, List, Set, Tuple, Optional
import pathlib

# 3rd party
import numpy as np

import h5py

from scipy.spatial import ConvexHull

# Our own imports
from ._grid_db import (
    calc_average_tri_density,
    calc_average_segment_angle,
    calc_average_segment_divergence,
)

from . import hdf5_cache
from ..utils import (
    calc_delaunay_adjacency, warp_to_circle, Hypermap, points_in_polygon,
    inv_warp_to_circle,
)
from ..tracking import calc_track_persistence, rolling_interp, rolling_window

# Constants

RADIAL_MIN = -1.1  # Minimum radius to sample
RADIAL_MAX = 1.1  # Maximum radius to sample
RADIAL_SAMPLES = 200  # How many samples to make for the radial interpolation

MAX_DISTANCE = 50  # um - max distance to make links along


# Classes


class GridDB(object):
    """ Convert tracks back to a space/time graph

    Loading and Saving:

    * :py:meth:`GridDB.from_hdf5`: Load the database from an HDF5 file
    * :py:meth:`GridDB.to_hdf5`: Save the database to an HDF5 file

    Main Pipeline (used by `calc_triangulated_stats.py`):

    * :py:meth:`GridDB.add_track`: Add a track to the database
    * :py:meth:`GridDB.triangulate_grid`: Calculate framewise triangulation
    * :py:meth:`GridDB.warp_grid_to_circle`: Warp the mesh to a normalized circle
    * :py:meth:`GridDB.calc_radial_stats`: Calculate track-wise statistics, binned by radius
    * :py:meth:`GridDB.calc_local_densities_mesh`: Calculate triangle-wise densities
    * :py:meth:`GridDB.calc_delta_divergence_mesh`: Calculate triangle-wise divergence
    * :py:meth:`GridDB.calc_delta_curl_mesh`: Calculate triangle-wise curl

    Query Methods:

    * :py:meth:`GridDB.get_timepoint_range`: Get the valid timepoints in this database
    * :py:meth:`GridDB.get_track_ids`: Get the track IDs in this database
    * :py:meth:`GridDB.get_track_values`: Get values along a track
    * :py:meth:`GridDB.interp_track_values`: Get and interpolate values along a track
    * :py:meth:`GridDB.get_track_summary`: Get summary statistics along a track
    * :py:meth:`GridDB.get_all_track_lengths`: Get the lengths for all track
    * :py:meth:`GridDB.get_flattened_values`: Get all the values in a mesh, flattened into an array
    * :py:meth:`GridDB.get_longest_perimeter`: Get the perimeter at a given timepoint

    Neighborhood search:

    * :py:meth:`GridDB.find_all_neighboring_track_timelines`: Collect all track neighbors along a timeline
    * :py:meth:`GridDB.find_neighboring_track_timelines`: Collect a single track's neighbors along a timeline
    * :py:meth:`GridDB.find_neighboring_tracks`: Find a single track's neighbors at a particular time
    * :py:meth:`GridDB.get_perimeter_point_ids`: Find the perimeter points for a given point set
    * :py:meth:`GridDB.get_perimeter_timeline_point_ids`: Find the perimeter point ids across time

    ROI Selection and subsetting:

    * :py:meth:`GridDB.find_coordinates_for_timepoint`: Load the coordinate array by timepoint
    * :py:meth:`GridDB.find_values_for_points`: Find all the values for a set of points
    * :py:meth:`GridDB.find_values_for_point_timeline`: Find all the values in a set of points over time
    * :py:meth:`GridDB.find_points_in_roi`: Find all points inside a region of interest at a timepoint
    * :py:meth:`GridDB.find_point_timelines_in_roi`: Find all points inside a region of interest over time
    * :py:meth:`GridDB.find_tracks_in_roi`: Find all tracks inside a region of interest at a timepoint
    * :py:meth:`GridDB.find_track_timelines_in_roi`: Find all tracks inside a region of interest over time
    * :py:meth:`GridDB.warp_to_timepoint`: Warp a set of points to the unit circle at a timepoint
    * :py:meth:`GridDB.inv_warp_to_timepoint`: Warp a set of points back from the unit circle at a timepoint

    Mesh helpers:

    * :py:meth:`GridDB.add_track_to_mesh`: Add a track to the forward and reverse mesh indices
    * :py:meth:`GridDB.reduce_mesh`: Average deltas from two meshes into a single mesh

    Initial arguments:

    :param int processes:
        Number of processes to use to generate tracks
    :param int max_timepoint:
        Maximum timepoint to process
    :param float space_scale:
        Scale factor for um/pixel
    :param float time_scale=1.0:
        Scale factor for minutes/frame
    :param float radial_min:
        Minimum radius to sample when gridding the images
    :param float radial_max:
        Maximum radius to sample when gridding the images
    :param float radial_samples:
        Number of points in x, y to generate an image grid
    """

    # Arrays are stored as n x k, where n is the number of points
    # This dictionary gives the value of k for different data types
    NUM_ARRAY_ARGS = {
        'timepoint_coords': 2,
        'timepoint_real_coords': 3,
        'timepoint_warp_coords': 2,
        'timepoint_warp_radius': 1,
        'local_densities_mesh': 1,
        'local_cell_areas_mesh': 1,
        'delta_curl_mesh': 1,
        'delta_divergence_mesh': 1,
        'local_displacement_mesh': 1,
        'local_distance_mesh': 1,
        'local_disp_vs_dist_mesh': 1,
        'local_velocity_mesh': 1,
        'local_speed_mesh': 1,
        'local_persistence_mesh': 1,
    }

    def __init__(self,
                 processes: int = 1,
                 max_timepoint: int = -1,
                 min_timepoint: int = 0,
                 space_scale: float = 1.0,
                 time_scale: float = 1.0,
                 radial_min: float = RADIAL_MIN,
                 radial_max: float = RADIAL_MAX,
                 radial_samples: int = RADIAL_SAMPLES,
                 min_persistence_points: int = 3,
                 resample_factor: int = 4,
                 interp_points: int = 7,
                 smooth_points: int = 0):

        self.min_timepoint = min_timepoint
        self.max_timepoint = max_timepoint
        self.processes = processes

        self.space_scale = space_scale
        self.time_scale = time_scale

        print('Space scale {:0.2f} um/px'.format(space_scale))
        print('Time scale {:0.2f} min/frame'.format(time_scale))

        # Parameters for warping meshes to a circle and gridding them
        self.radial_min = radial_min
        self.radial_max = radial_max
        self.radial_samples = radial_samples

        self.min_persistence_points = min_persistence_points
        self.resample_factor = resample_factor
        self.interp_points = interp_points
        self.smooth_points = smooth_points

        self.timepoint_coords = {}
        self.timepoint_links = {}

        # Link of track index to time index: point index
        self.track_links = {}
        self.track_links_inv = {}

        self.timepoint_meshes = {}
        self.timepoint_triangles = {}
        self.timepoint_perimeters = {}

        self.timepoint_real_coords = {}
        self.timepoint_warp_coords = {}
        self.timepoint_warp_radius = {}

        self.track_peristences = {}

        # New field attributes
        self.local_densities_mesh = {}
        self.local_cell_areas_mesh = {}
        self.delta_curl_mesh = {}
        self.delta_divergence_mesh = {}

        self.local_displacement_mesh = {}
        self.local_distance_mesh = {}
        self.local_disp_vs_dist_mesh = {}

        self.local_velocity_mesh = {}
        self.local_speed_mesh = {}
        self.local_persistence_mesh = {}

    @classmethod
    def from_hdf5(cls, infile, lazy=False, **kwargs):
        """ Load the data from an HDF5 file

        :param Path infile:
            The HDF5 file to load
        :returns:
            The grid database
        """
        if isinstance(infile, (str, pathlib.Path)):
            db = h5py.File(str(infile), 'r')
            needs_closing = not lazy
        else:
            db = infile
            needs_closing = False

        try:
            obj = cls(**kwargs)

            # Copy attributes
            obj.time_scale = db.attrs['time_scale']
            obj.space_scale = db.attrs['space_scale']

            # Copy the simple track linkage
            print('Loading single tracks...')
            cache = hdf5_cache.HDF5CoordCache('timepoint_coords')
            obj.timepoint_coords = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordCache('timepoint_real_coords')
            obj.timepoint_real_coords = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5TimepointLinkCache('timepoint_links')
            obj.timepoint_links = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5TrackLinkCache('track_links')
            obj.track_links = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5TrackLinkInvCache('track_links')
            obj.track_links_inv = cache.from_hdf5(db, lazy=lazy)

            # Copy the spatial linkage
            print('Loading meshes...')
            cache = hdf5_cache.HDF5TimepointMeshCache('timepoint_meshes')
            obj.timepoint_meshes = cache.from_hdf5(db, lazy=lazy)

            # Copy the triangles
            cache = hdf5_cache.HDF5TimepointTriangleCache('timepoint_triangles')
            obj.timepoint_triangles = cache.from_hdf5(db, lazy=lazy)

            # Copy the perimeters
            cache = hdf5_cache.HDF5TimepointPerimeterCache('timepoint_perimeters')
            obj.timepoint_perimeters = cache.from_hdf5(db, lazy=lazy)

            # Load the warp data
            cache = hdf5_cache.HDF5CoordCache('timepoint_warp_coords')
            obj.timepoint_warp_coords = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('timepoint_warp_radius')
            obj.timepoint_warp_radius = cache.from_hdf5(db, lazy=lazy)

            # Load the persistence stats
            cache = hdf5_cache.HDF5PersistenceCache('track_peristences')
            obj.track_peristences = cache.from_hdf5(db, lazy=lazy)

            # Load the density stats
            print('Loading density parameters...')
            cache = hdf5_cache.HDF5CoordValueCache('local_densities_mesh')
            obj.local_densities_mesh = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('local_cell_areas_mesh')
            obj.local_cell_areas_mesh = cache.from_hdf5(db, lazy=lazy)

            # Load the velocity stats
            print('Loading velocity parameters...')
            cache = hdf5_cache.HDF5CoordValueCache('local_displacement_mesh')
            obj.local_displacement_mesh = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('local_distance_mesh')
            obj.local_distance_mesh = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('local_disp_vs_dist_mesh')
            obj.local_disp_vs_dist_mesh = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('local_speed_mesh')
            obj.local_speed_mesh = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('local_velocity_mesh')
            obj.local_velocity_mesh = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('local_persistence_mesh')
            obj.local_persistence_mesh = cache.from_hdf5(db, lazy=lazy)

            # Load the delta maps
            print('Loading delta parameters...')
            cache = hdf5_cache.HDF5CoordValueCache('delta_divergence_mesh')
            obj.delta_divergence_mesh = cache.from_hdf5(db, lazy=lazy)

            cache = hdf5_cache.HDF5CoordValueCache('delta_curl_mesh')
            obj.delta_curl_mesh = cache.from_hdf5(db, lazy=lazy)
        finally:
            if needs_closing:
                db.close()
        return obj

    def to_hdf5(self, outfile):
        """ Save the data to an HDF5 file

        :param Path outfile:
            The HDF5 file to save to
        """
        if isinstance(outfile, (str, pathlib.Path)):
            outfile = pathlib.Path(outfile)
            outfile.parent.mkdir(parents=True, exist_ok=True)

            db = h5py.File(str(outfile), 'w')
            needs_closing = True
        else:
            db = outfile
            needs_closing = False

        try:
            db.attrs['time_scale'] = self.time_scale
            db.attrs['space_scale'] = self.space_scale

            # Save the timepoint coordinate index
            cache = hdf5_cache.HDF5CoordCache('timepoint_coords')
            cache.to_hdf5(db, self.timepoint_coords)

            cache = hdf5_cache.HDF5CoordCache('timepoint_real_coords')
            cache.to_hdf5(db, self.timepoint_real_coords)

            # Save the timepoint link index
            cache = hdf5_cache.HDF5TimepointLinkCache('timepoint_links')
            cache.to_hdf5(db, self.timepoint_links)

            # Save the track link index
            cache = hdf5_cache.HDF5TrackLinkCache('track_links')
            cache.to_hdf5(db, self.track_links)

            # Save the mesh database
            cache = hdf5_cache.HDF5TimepointMeshCache('timepoint_meshes')
            cache.to_hdf5(db, self.timepoint_meshes)

            cache = hdf5_cache.HDF5TimepointTriangleCache('timepoint_triangles')
            cache.to_hdf5(db, self.timepoint_triangles)

            cache = hdf5_cache.HDF5TimepointPerimeterCache('timepoint_perimeters')
            cache.to_hdf5(db, self.timepoint_perimeters)

            # Save the warp to circle data
            cache = hdf5_cache.HDF5CoordCache('timepoint_warp_coords')
            cache.to_hdf5(db, self.timepoint_warp_coords)

            cache = hdf5_cache.HDF5CoordValueCache('timepoint_warp_radius')
            cache.to_hdf5(db, self.timepoint_warp_radius)

            # Save the persistence stats
            cache = hdf5_cache.HDF5PersistenceCache('track_peristences')
            cache.to_hdf5(db, self.track_peristences)

            # Save the fields
            print('Saving density parameters...')
            cache = hdf5_cache.HDF5CoordValueCache('local_cell_areas_mesh')
            cache.to_hdf5(db, self.local_cell_areas_mesh)

            cache = hdf5_cache.HDF5CoordValueCache('local_densities_mesh')
            cache.to_hdf5(db, self.local_densities_mesh)

            print('Saving velocity parameters...')
            cache = hdf5_cache.HDF5CoordValueCache('local_displacement_mesh')
            cache.to_hdf5(db, self.local_displacement_mesh)

            cache = hdf5_cache.HDF5CoordValueCache('local_distance_mesh')
            cache.to_hdf5(db, self.local_distance_mesh)

            cache = hdf5_cache.HDF5CoordValueCache('local_disp_vs_dist_mesh')
            cache.to_hdf5(db, self.local_disp_vs_dist_mesh)

            cache = hdf5_cache.HDF5CoordValueCache('local_velocity_mesh')
            cache.to_hdf5(db, self.local_velocity_mesh)

            cache = hdf5_cache.HDF5CoordValueCache('local_speed_mesh')
            cache.to_hdf5(db, self.local_speed_mesh)

            cache = hdf5_cache.HDF5CoordValueCache('local_persistence_mesh')
            cache.to_hdf5(db, self.local_persistence_mesh)

            # Load the delta maps
            print('Saving delta parameters...')
            cache = hdf5_cache.HDF5CoordValueCache('delta_divergence_mesh')
            cache.to_hdf5(db, self.delta_divergence_mesh)

            cache = hdf5_cache.HDF5CoordValueCache('delta_curl_mesh')
            cache.to_hdf5(db, self.delta_curl_mesh)
        finally:
            if needs_closing:
                db.close()

    # Loader for multiple attributes from the lazy cache

    def load(self, *args):
        """ Actually load values from the lazy cache """
        # Handling loading from a list or tuple too
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        # Now for each attribute, load it, if necessary
        for attr in args:
            value = getattr(self, attr)
            if hasattr(value, 'load'):
                print('Loading lazy attr "{}"'.format(attr))
                setattr(self, attr, value.load())

    # Main methods

    def add_track(self, track):
        """ Add a track to the database

        :param Link track:
            The track to add to the grid
        """
        space_scale = self.space_scale
        time_scale = self.time_scale

        trackidx = len(self.track_links)

        pt, px, py = track.to_arrays()

        assert pt.shape == px.shape
        assert pt.shape == py.shape

        for ist in range(pt.shape[0]-1):
            t1, t2 = pt[ist:ist+2]
            x1, x2 = px[ist:ist+2]
            y1, y2 = py[ist:ist+2]

            # Only add the old coordinates in the first step
            if ist == 0:
                self.timepoint_coords.setdefault(t1, []).append((x1, y1))
                self.timepoint_real_coords.setdefault(t1, []).append(
                    (x1 * space_scale, y1 * space_scale, t1 * time_scale))

            # Always add the second coordinates
            self.timepoint_coords.setdefault(t2, []).append((x2, y2))
            self.timepoint_real_coords.setdefault(t2, []).append(
                (x2 * space_scale, y2 * space_scale, t2 * time_scale))

            idx1 = len(self.timepoint_coords[t1])-1
            idx2 = len(self.timepoint_coords[t2])-1

            link = self.timepoint_links.setdefault((t1, t2), {})
            assert idx1 not in link
            link[idx1] = idx2

            # Maintain a mapping of track link index to individual time/point coordinates
            track_link = self.track_links.setdefault(trackidx, {})
            track_link[t1] = idx1
            track_link[t2] = idx2

            # And maintain the inverted index to map point indicies back to tracks
            track_link_inv = self.track_links_inv.setdefault(t1, {})
            track_link_inv[idx1] = trackidx
            track_link_inv = self.track_links_inv.setdefault(t2, {})
            track_link_inv[idx2] = trackidx

    def triangulate_grid(self, max_distance: float = MAX_DISTANCE):
        """ Convert the points to a mesh

        :param float max_distance:
            Max distance to link triangles in um
        """
        print('Triangulating...')
        max_distance_px = max_distance / self.space_scale
        print('Max distance for triangulation: {:0.1f} um = {:0.1f} px'.format(max_distance, max_distance_px))
        items = [(timepoint, max_distance_px) for timepoint in self.get_timepoint_range()]

        with Hypermap(processes=self.processes) as pool:
            res = pool.map(self._triangulate_timepoint, items)
            for timepoint, links, tris, perimeters in res:
                assert timepoint not in self.timepoint_meshes
                self.timepoint_meshes[timepoint] = links
                self.timepoint_triangles[timepoint] = tris
                self.timepoint_perimeters[timepoint] = perimeters

    def warp_grid_to_circle(self):
        """ Warp the grids to the unit circle """

        items = self.get_timepoint_range()
        print('Warping to circular coordinate frame...')

        with Hypermap(processes=self.processes) as pool:
            res = pool.map(self._warp_timepoint_to_circle, items)
            for timepoint, warp_coords in res:
                if warp_coords is None:
                    warp_coords = np.full((len(self.timepoint_coords[timepoint]), 2), fill_value=np.nan)

                assert len(warp_coords) == len(self.timepoint_coords[timepoint])
                assert timepoint not in self.timepoint_warp_coords
                assert timepoint not in self.timepoint_warp_radius

                # Cache the radial position for easy binning
                warp_coords = [tuple(c) for c in warp_coords]
                warp_radius = [np.sqrt(x**2 + y**2) for x, y in warp_coords]

                self.timepoint_warp_coords[timepoint] = warp_coords
                self.timepoint_warp_radius[timepoint] = warp_radius

    def calc_radial_stats(self):
        """ Bin the data by radial distance """

        print('Calculating radial stats...')

        local_displacement_mesh = {}
        local_distance_mesh = {}
        local_disp_vs_dist_mesh = {}

        local_velocity_mesh = {}
        local_speed_mesh = {}
        local_persistence_mesh = {}

        items = self.get_track_ids()

        with Hypermap(processes=self.processes) as pool:
            for trackidx, res in pool.map(self._calc_track_persistence, items):
                self.track_peristences[trackidx] = res

                sm_tt = res.tt
                sm_xx = res.xx
                sm_yy = res.yy
                sm_state = res.mask
                tidx = res.tidx
                xidx = res.xidx

                dx = sm_xx[1:] - sm_xx[:-1]
                dy = sm_yy[1:] - sm_yy[:-1]
                ds = np.sqrt(dx**2 + dy**2)

                displacement = np.sqrt((sm_xx[1:] - sm_xx[0])**2 + (sm_yy[1:] - sm_yy[0])**2)
                distance = np.cumsum(ds)
                tot_time = sm_tt[1:] - sm_tt[0]

                self.add_track_to_mesh(local_displacement_mesh, tidx, xidx, sm_tt[:-1], displacement)
                self.add_track_to_mesh(local_distance_mesh, tidx, xidx, sm_tt[:-1], distance)
                self.add_track_to_mesh(local_disp_vs_dist_mesh, tidx, xidx, sm_tt[:-1], displacement/distance)

                self.add_track_to_mesh(local_velocity_mesh, tidx, xidx, sm_tt[:-1], displacement/tot_time)
                self.add_track_to_mesh(local_speed_mesh, tidx, xidx, sm_tt[:-1], distance/tot_time)

                self.add_track_to_mesh(local_persistence_mesh, tidx, xidx, sm_tt, sm_state)

        self.local_displacement_mesh = local_displacement_mesh
        self.local_distance_mesh = local_distance_mesh
        self.local_disp_vs_dist_mesh = local_disp_vs_dist_mesh
        self.local_velocity_mesh = local_velocity_mesh
        self.local_speed_mesh = local_speed_mesh
        self.local_persistence_mesh = local_persistence_mesh

    def calc_local_densities_mesh(self):
        """ Calculate the density as a mesh """

        items = self.get_timepoint_range()

        local_densities_mesh = {}
        local_cell_areas_mesh = {}
        print('Calculating density mesh...')
        with Hypermap(processes=self.processes) as pool:
            for timepoint, res in pool.map(self._calc_local_density_mesh, items):
                if res is None:
                    res = np.full((len(self.timepoint_coords[timepoint]), ), fill_value=np.nan)
                local_cell_areas_mesh[timepoint] = list(res)
                local_densities_mesh[timepoint] = list(1.0 / res)
        self.local_cell_areas_mesh = local_cell_areas_mesh
        self.local_densities_mesh = local_densities_mesh

    def calc_delta_divergence_mesh(self):
        """ Calculate the divergence as the delta between the two densities """

        items = self.get_timepoint_range()

        # Map step - calculate mid-timepoint values
        delta_divergence_mesh_map = {}
        print('Calculating divergence mesh...')
        with Hypermap(processes=self.processes) as pool:
            for t0, t1, res in pool.map(self._calc_delta_divergence_mesh, items):
                if res is None:
                    res = np.full((len(self.timepoint_coords[t1]), ), fill_value=np.nan)
                delta_divergence_mesh_map[t0, t1] = res

        # Reduce step, convert pairs back to single points
        print('Averaging divergence mesh')
        self.delta_divergence_mesh = self.reduce_mesh(delta_divergence_mesh_map, items)

    def calc_delta_curl_mesh(self):
        """ Calculate the divergence as the delta between the two densities """

        items = self.get_timepoint_range()

        print('Calculating curl mesh...')
        delta_curl_mesh_map = {}
        with Hypermap(processes=self.processes) as pool:
            for t0, t1, res in pool.map(self._calc_delta_curl_mesh, items):
                if res is None:
                    res = np.full((len(self.timepoint_coords[t1]), ), fill_value=np.nan)
                delta_curl_mesh_map[t0, t1] = res
        print('Averaging curl mesh')
        self.delta_curl_mesh = self.reduce_mesh(delta_curl_mesh_map, items)

    # Helper methods

    def get_perimeter_point_ids(self,
                                timepoint: int,
                                point_ids: List[int],
                                use_mesh: str = 'timepoint_coords') -> List[int]:
        """ Get the perimeter point ids, given a list of point ids and a timepoint

        :param int timepoint:
            The timepoint to load
        :param List[int] point_ids:
            The point ids to load
        :param str use_mesh:
            Use this mesh when calculating the perimeter ids
        :returns:
            Those point ids corresponding to points on the perimeter
        """
        mesh = self.find_coordinates_for_timepoint(use_mesh, timepoint)
        points = np.array([mesh[i] for i in point_ids])
        hull = ConvexHull(points)
        return [point_ids[i] for i in hull.vertices]

    def get_perimeter_timeline_point_ids(self,
                                         point_ids: Dict[int, List[int]],
                                         use_mesh: str = 'timepoint_coords') -> Dict[int, List[int]]:
        """ Get the perimeter point ids for a set of timepoints

        :param Dict[int, List[int]] point_ids:
            The timepoint: List[point ids to load] mapping
        :param str use_mesh:
            Use this mesh when calculating the perimeter ids
        :returns:
            Those point ids corresponding to points on the perimeter for each timepoint
        """
        perimeter_ids = {}
        for timepoint, ids in point_ids.items():
            perimeter_ids[timepoint] = self.get_perimeter_point_ids(
                timepoint, ids, use_mesh=use_mesh)
        return perimeter_ids

    def warp_to_timepoint(self, timepoint: int,
                          coords: np.ndarray,
                          i_max: Optional[int] = None,
                          r_max: Optional[float] = None) -> np.ndarray:
        """ Warp a set of points to the unit circle at a timepoint

        :param int timepoint:
            The timepoint to warp to
        :param ndarray coords:
            The n x 2 array of coordinates to warp
        :returns:
            The n x 2 array of warped coordinates
        """
        perimeter = self.get_longest_perimeter(timepoint)
        if perimeter is None:
            return None
        if i_max is None:
            i_max = self.radial_samples
        if r_max is None:
            r_max = -1
        return warp_to_circle(coords, perimeter, i_max=i_max, r_max=r_max)

    def inv_warp_to_timepoint(self, timepoint: int,
                              coords: np.ndarray,
                              i_max: Optional[int] = None) -> np.ndarray:
        """ Warp a set of points back from the unit circle at a timepoint

        :param int timepoint:
            The timepoint to warp from
        :param ndarray coords:
            The n x 2 array of warped coordinates
        :returns:
            The n x 2 array of inverse warped coordinates
        """
        perimeter = self.get_longest_perimeter(timepoint)
        if perimeter is None:
            return None
        if i_max is None:
            i_max = self.radial_samples
        return inv_warp_to_circle(coords, perimeter, i_max=i_max)

    def invert_track_ids(self,
                         track_ids: List[int]) -> List[int]:
        """ Invert a set of track ids

        :param List[int] track_ids:
            The set of track ids to invert
        :returns:
            All track ids not in that set
        """
        track_ids = set(track_ids)
        return [i for i in self.get_track_ids() if i not in track_ids]

    def invert_point_ids(self,
                         point_ids: List[int],
                         timepoint: int) -> List[int]:
        """ Invert a set of point ids at a timepoint

        :param List[int] point_ids:
            The set of point ids to invert
        :param int timepoint:
            The timepoint to invert the ids
        :returns:
            All point ids not in that set at that time
        """
        point_ids = set(point_ids)
        return [i for i in range(len(self.timepoint_coords[timepoint]))
                if i not in point_ids]

    def invert_point_timeline_ids(self, *args) -> Dict[int, List[int]]:
        """ Invert a set of point ids over time

        :param \\*args:
            The list of dictionaries of timepoint: point ids to invert
        :returns:
            A dictionary containing inverted points at all times
        """
        all_point_ids = {}
        for point_timeline_ids in args:
            for timepoint, point_ids in point_timeline_ids.items():
                all_point_ids.setdefault(timepoint, set()).update(point_ids)
        inv_point_ids = {}
        for timepoint, point_ids in all_point_ids.items():
            inv_point_ids[timepoint] = [i for i in range(len(self.timepoint_coords[timepoint]))
                                        if i not in point_ids]
        return inv_point_ids

    def find_track_timelines_in_roi(self,
                                    perimeter: np.ndarray,
                                    timepoints: Optional[List[int]] = None,
                                    use_mesh: str = 'timepoint_coords') -> List[int]:
        """ Find all the tracks inside an ROI over time

        :param ndarray perimeter:
            The n x 2 array of perimeter coordinates for the polygon
        :param str use_mesh:
            Which coordinate mesh to use ('coords', 'warp_coords', 'real_coords')
        :returns:
            A list of track ids inside this ROI
        """
        if timepoints is None:
            timepoints = self.get_timepoint_range()

        track_ids = set()
        for timepoint in timepoints:
            track_ids.update(self.find_tracks_in_roi(perimeter, timepoint, use_mesh=use_mesh))
        return list(sorted(track_ids))

    def find_tracks_in_roi(self,
                           perimeter: np.ndarray,
                           timepoint: int,
                           use_mesh: str = 'timepoint_coords') -> List[int]:
        """ Find all the tracks inside an ROI at a single timepoint

        :param ndarray perimeter:
            The n x 2 array of perimeter coordinates for the polygon
        :param int timepoint:
            The timepoint to search for tracks in
        :param str use_mesh:
            Which coordinate mesh to use ('coords', 'warp_coords', 'real_coords')
        :returns:
            A list of track ids inside this ROI
        """
        point_ids = self.find_points_in_roi(perimeter=perimeter,
                                            timepoint=timepoint,
                                            use_mesh=use_mesh)
        return [self.track_links_inv[timepoint][point_id]
                for point_id in point_ids]

    def find_coordinates_for_timepoint(self, mesh: str, timepoint: int) -> List[Tuple]:
        """ Find all the coordinates for a mesh at a timepoint

        :param str mesh:
            The mesh to load
        :param int timepoint:
            The timepoint to load
        :returns:
            A list of (x, y) coordinates at that timepoint
        """
        mesh = mesh.lower()
        # Unpack the mesh and
        if mesh in ('coords', 'timepoint_coords'):
            points = self.timepoint_coords[timepoint]
        elif mesh in ('warp', 'warp_coords', 'timepoint_warp_coords'):
            points = self.timepoint_warp_coords[timepoint]
        elif mesh in ('real', 'real_coords', 'timepoint_real_coords'):
            points = [(x, y) for x, y, _ in self.timepoint_real_coords[timepoint]]
        else:
            raise KeyError('Unknown mesh type: "{}"'.format(mesh))
        return points

    def find_points_in_roi(self,
                           perimeter: np.ndarray,
                           timepoint: int,
                           use_mesh: str = 'timepoint_coords') -> List[int]:
        """ Find all the points inside an ROI at a single timepoint

        :param ndarray perimeter:
            The n x 2 array of perimeter coordinates for the polygon
        :param int timepoint:
            The timepoint to search for tracks in
        :param str use_mesh:
            Which coordinate mesh to use ('coords', 'warp_coords', 'real_coords')
        :returns:
            A list of point ids inside this ROI
        """
        points = self.find_coordinates_for_timepoint(use_mesh, timepoint)
        if len(points) < 1:
            return []

        point_ids = np.array(list(range(len(points))))
        point_mask = points_in_polygon(perimeter, np.array(points))
        return list(point_ids[point_mask])

    def find_point_timelines_in_roi(self,
                                    perimeter: np.ndarray,
                                    timepoints: Optional[List[int]] = None,
                                    use_mesh: str = 'timepoint_coords') -> Dict[int, List[int]]:
        """ Find all the points inside an ROI over time

        :param ndarray perimeter:
            The n x 2 array of perimeter coordinates for the polygon
        :param str use_mesh:
            Which coordinate mesh to use ('coords', 'warp_coords', 'real_coords')
        :returns:
            A dictionary mapping timepoint ids to a list of point ids per timepoint
        """
        if timepoints is None:
            timepoints = self.get_timepoint_range()
        points = {}
        for timepoint in timepoints:
            points[timepoint] = self.find_points_in_roi(
                perimeter=perimeter, timepoint=timepoint, use_mesh=use_mesh)
        return points

    def find_values_for_points(self,
                               field: str,
                               points: List[int],
                               timepoint: int) -> np.ndarray:
        """ Find all the values under a point set

        :param str field:
            Which field to get values for, with k values per point
        :param List[int] points:
            Which n-point ids to select
        :param int timepoint:
            Which timepoint to select
        :returns:
            An array of values for that point subset, of size n x k.
        """
        mesh = getattr(self, field)[timepoint]
        return np.array([mesh[i] for i in points])

    def find_values_for_point_timeline(self,
                                       field: str,
                                       points: Dict[int, List[int]]) -> Dict[int, np.ndarray]:
        """ Find all the values under a point timeline

        :param str field:
            Which field to get values for, with k values per point
        :param Dict[int, List[int]] points:
            Which n-point ids to select
        :returns:
            A dictionary of values at each selected timepoint
        """
        values = {}
        for timepoint, point_set in points.items():
            values[timepoint] = self.find_values_for_points(
                field=field, points=point_set, timepoint=timepoint)
        return values

    def get_timepoint_range(self) -> List[int]:
        """ Get the valid set of timepoints for interpolation

        :returns:
            The list of timepoints to generate grids over
        """
        # Find a loaded coordinate database to get the time keys off of
        coord_attrs = [
            'timepoint_coords', 'timepoint_real_coords',
            'timepoint_warp_coords', 'timepoint_warp_radius',
        ]
        all_timepoints = None
        for attr in coord_attrs:
            value = getattr(self, attr)
            if isinstance(value, hdf5_cache.LazyLoader):
                continue
            all_timepoints = value.keys()
            break
        if all_timepoints is None:
            raise KeyError('No coordinate attributes loaded: {}'.format(coord_attrs))
        min_timepoint = self.min_timepoint if self.min_timepoint is not None else -1
        max_timepoint = self.max_timepoint if self.max_timepoint is not None else -1
        return [timepoint for timepoint in sorted(all_timepoints)
                if ((min_timepoint < 0 or timepoint >= min_timepoint) and
                    (max_timepoint < 0 or timepoint <= max_timepoint))]

    def get_track_ids(self) -> List[int]:
        """ Get all the valid track identifiers

        :returns:
            The list of track ids for iteration
        """
        return list(sorted(self.track_links.keys()))

    def add_track_to_mesh(self,
                          mesh: Dict,
                          tidx: List[int],
                          xidx: List[int],
                          track_time: List[float],
                          track_values: List[float]):
        """ Add track-wise data to a mesh

        :param dict mesh:
            The mesh to add track data to (modifies the dictionary!)
        :param list tidx:
            The time indicies to write to the mesh
        :param list xidx:
            The space indicies to write to the mesh
        :param list track_time:
            The real time values along the track
        :param list track_values:
            The values at those time points along the track
        """
        track_time = np.array(track_time)
        track_values = np.array(track_values)

        for t0, x0 in zip(tidx, xidx):
            timepoint_mesh = mesh.setdefault(t0, [])
            if len(timepoint_mesh) <= x0:
                timepoint_mesh.extend([np.nan] * (x0 - len(timepoint_mesh) + 1))

            t_low = t0 * self.time_scale
            t_high = (t0 + 1) * self.time_scale

            t_mask = np.logical_and(track_time >= t_low,
                                    track_time < t_high)
            if ~np.any(t_mask):
                timepoint_mesh[x0] = np.nan
            else:
                timepoint_mesh[x0] = np.nanmean(track_values[t_mask])

    def get_track_values(self, trackidx: int, *args) -> List[np.ndarray]:
        """ Get values along a track

        :param int trackidx:
            The track index to load
        :param str \\*args:
            The individual grid databases to load
        :returns:
            A list of time index, space index, followed by n tracks, one for each argument
        """
        # Load the (time, index) maps for this track
        indices = self.track_links[trackidx]

        # Store everything in a list of tracks, first two are tidx and xidx
        tracks = [[], []]

        # Cache the track index, attributes
        attrs = []

        # Initialize all the tracks
        ct = 2
        for arg in args:
            num = self.NUM_ARRAY_ARGS[arg]
            attrs.append((ct, num, getattr(self, arg)))
            for _ in range(num):
                tracks.append([])
            ct += num

        # Assemble the tracks, one for each argument
        for t0, i0 in sorted(indices.items()):
            tracks[0].append(t0)
            tracks[1].append(i0)

            for ct, num, attr in attrs:
                if t0 not in attr:
                    for j in range(num):
                        tracks[ct + j].append(np.nan)
                else:
                    vals = attr[t0][i0]
                    if num == 1:
                        vals = [vals]
                    for j, val in enumerate(vals):
                        tracks[ct + j].append(val)
        return [np.array(t) for t in tracks]

    def interp_track_values(self, trackidx: int, *args, **kwargs) -> List[np.ndarray]:
        """ Interpolate values over a track

        :param int trackidx:
            The track index to load
        :param str \\*args:
            The individual grid databases to load
        :returns:
            The time index, then the tracks, interpolated over those values
        """
        resample_factor = kwargs.get('resample_factor', self.resample_factor)
        interp_points = kwargs.get('interp_points', self.interp_points)
        smooth_points = kwargs.get('smooth_points', self.smooth_points)

        values = self.get_track_values(trackidx, *args)
        tidx = values[0]
        values = values[2:]

        # Interpolate over an expanded time vector
        sm_tt = np.linspace(np.min(tidx), np.max(tidx), resample_factor*tidx.shape[0])
        smooth_values = [sm_tt]
        for value in values:
            value = rolling_interp(sm_tt, tidx, value, interp_points, order=1)
            if smooth_points > 0:
                value = np.mean(rolling_window(value, smooth_points*resample_factor, pad='same'), 1)
            smooth_values.append(value)
        return smooth_values

    def get_track_summary(self, trackidx: int, *args, **kwargs) -> List[float]:
        """ Get a summary statistic for each attribute in a track

        :param int trackidx:
            The track index to load
        :param str \\*args:
            The individual grid databases to load
        :param str func:
            Which function to apply (default: mean)
        :returns:
            A list of values, one for each attribute
        """
        funcs = {
            'absmean': lambda x: np.nanmean(np.abs(x)),
            'mean': np.nanmean,
            'std': np.nanstd,
            'median': np.nanmedian,
            'absmedian': lambda x: np.nanmedian(np.abs(x)),
            'min': np.nanmin,
            'max': np.nanmax,
            'count': lambda x: np.sum(~np.isnan(x)),
        }
        func = funcs[kwargs.get('func', 'mean')]

        values = []
        # Skip the first two values because they are the time and space indicies
        for res in self.get_track_values(trackidx, *args)[2:]:
            values.append(func(res))
        return values

    def get_all_track_lengths(self) -> Dict[int, int]:
        """ Get the lengths for all the tracks

        :returns:
            A dictionary mapping track index to track length
        """
        return {k: len(v) for k, v in self.track_links.items()}

    def get_longest_perimeter(self, timepoint: int) -> np.ndarray:
        """ Get the longest perimeter for a timepoint

        :param int timepoint:
            The timepoint to load
        :returns:
            The coordinates of the longest perimeter
        """

        points = np.array(self.timepoint_coords[timepoint])
        perimeters = self.timepoint_perimeters[timepoint]

        best_perimeter = None
        perimeter_len = 0
        for perimeter in perimeters:
            if len(perimeter) > perimeter_len:
                best_perimeter = perimeter
                perimeter_len = len(perimeter)
        if best_perimeter is None:
            return None

        best_perimeter = np.stack([points[i, :] for i in best_perimeter], axis=0)
        assert best_perimeter.shape[1] == 2
        return best_perimeter

    def get_all_track_summaries(self, *args, **kwargs) -> List[np.ndarray]:
        """ Get the summary statistics over each track

        :param \\*args:
            Arguments to pass to :py:meth:`get_track_summaries`
        :param \\**kwargs:
            Keyword arguments to pass to :py:meth:`get_track_summaries`
        :returns:
            A List of numpy arrays of values, one for each track
        """

        # Collect the summary stats into arrays, one array per attribute
        summaries = {}
        for i in self.get_track_ids():
            for j, res in enumerate(self.get_track_summary(i, *args, **kwargs)):
                summaries.setdefault(j, []).append(res)
        return [np.array(summaries[k]) for k in sorted(summaries.keys())]

    def find_all_neighboring_track_timelines(self,
                                             trackidxs: Optional[List[int]] = None,
                                             distance: int = 1) -> Dict[int, Set[int]]:
        """ Find all the neighbors, for a set of tracks, at all times

        :param list[int] trackidxs:
            List of tracks to search along
        :param int distance:
            The distance, in number of links, to find neighbors at
        :returns:
            A dictionary mapping track indicies to sets of neighbors
        """
        if trackidxs is None:
            trackidxs = list(self.track_links)

        all_neighbors = {}
        for trackidx in trackidxs:
            all_neighbors[trackidx] = self.find_neighboring_track_timelines(
                trackidx=trackidx, distance=distance)
        return all_neighbors

    def find_neighboring_track_timelines(self,
                                         trackidx: int,
                                         distance: int = 1) -> Set[int]:
        """ Find all neighbors for a given track, over the entire track timeline

        :param int trackidx:
            Track to search along
        :param int distance:
            The distance, in number of links, to find neighbors at
        :returns:
            A set of nearest neighbor track indices for this track, across all time
        """
        all_neighbors = set()
        for timepoint in self.track_links[trackidx]:
            all_neighbors.update(self.find_neighboring_tracks(trackidx, timepoint, distance=distance))
        return all_neighbors

    def find_neighboring_tracks(self,
                                trackidx: int,
                                timepoint: int,
                                distance: int = 1) -> Set[int]:
        """ Find all neighbors for a given track, at a given timepoint

        :param int trackidx:
            Track to search along
        :param int timepoint:
            Timepoint to search at
        :param int distance:
            The distance, in number of links, to find neighbors at
        :returns:
            A set of nearest neighbor track indices for this track, at this time
        """
        if distance < 1:
            return {trackidx}

        # Remember all the tracks we've seen so far
        seen = {trackidx}
        frontier = {trackidx}
        for dist in range(1, distance + 1):
            if not frontier:
                break

            # Collect all the neighbors into a new frontier
            all_neighbors = set()
            for tidx in frontier:
                if tidx not in self.track_links:
                    continue
                if timepoint not in self.track_links[tidx]:
                    continue
                # Work out the point index for this track
                point_idx = self.track_links[tidx][timepoint]

                # Work out the neighbors in point space
                if timepoint not in self.timepoint_meshes:
                    continue
                if point_idx not in self.timepoint_meshes[timepoint]:
                    continue
                point_neighbors = self.timepoint_meshes[timepoint][point_idx]

                # Convert back to track space
                neighbors = (self.track_links_inv[timepoint][n] for n in point_neighbors)
                all_neighbors.update(n for n in neighbors if n not in seen)

            # Memorize the seen tracks and set the frontier to the next area
            seen.update(all_neighbors)
            frontier = all_neighbors

        return frontier

    def get_timepoint_values(self, timepoint: int, *args) -> List[np.ndarray]:
        """ Get flattened arrays at a specific timepoint

        :param int timepoint:
            The timepoint to collect
        :param str \\*args:
            The individual grid databases to load
        :returns:
            A list of the values for all those arrays at that timepoint
        """
        # Store everything in a list of tracks
        flat_values = []

        # Cache the track index, attributes
        attrs = []

        # Initialize all the tracks
        ct = 0
        for arg in args:
            num = self.NUM_ARRAY_ARGS[arg]
            attrs.append((ct, num, getattr(self, arg)[timepoint]))
            for _ in range(num):
                flat_values.append([])
            ct += num

        # Load and flatten all the requested values
        for ct, num, timepoint_values in attrs:
            if num == 1:
                timepoint_values = [(v, ) for v in timepoint_values]
            untimepoint_values = zip(*timepoint_values)
            for j, v in enumerate(untimepoint_values):
                flat_values[ct + j].extend(v)
        return [np.array(v) for v in flat_values]

    def get_flattened_values(self, *args) -> List[np.ndarray]:
        """ Get flattened arrays

        :param str \\*args:
            The individual grid databases to load
        :returns:
            A list of flattened arrays
        """
        # Store everything in a list of tracks
        flat_values = []

        # Cache the track index, attributes
        attrs = []

        # Initialize all the tracks
        ct = 0
        for arg in args:
            num = self.NUM_ARRAY_ARGS[arg]
            attrs.append((ct, num, getattr(self, arg)))
            for _ in range(num):
                flat_values.append([])
            ct += num

        # Load and flatten all the requested values
        for ct, num, attr in attrs:
            for timepoint_values in attr.values():
                if num == 1:
                    timepoint_values = [(v, ) for v in timepoint_values]
                untimepoint_values = zip(*timepoint_values)
                for j, v in enumerate(untimepoint_values):
                    flat_values[ct + j].extend(v)
        return [np.array(v) for v in flat_values]

    def reduce_mesh(self, mesh: Dict, timepoints: List[int]) -> Dict[int, list]:
        """ Reduce a mesh of forward and backwards differences to central differences

        :param dict mesh:
            The dictionary of (t1, t2): (v1, v2) to reduce
        :param list timepoints:
            The list of timepoints to merge
        :returns:
            A dictionary of timepoint: value for each point in the mesh
        """

        final_mesh = {}
        for t1 in timepoints:
            t0 = t1 - 1
            t2 = t1 + 1

            # Backwards divergence
            values01 = mesh.get((t0, t1), (None, None))[1]
            # Forwards divergence
            values12 = mesh.get((t1, t2), (None, None))[0]

            # Average the whole divergence vector
            if values01 is None and values12 is None:
                continue
            if values01 is None:
                values = values12
            elif values12 is None:
                values = -values01
            else:
                assert values01.shape == values12.shape
                values = np.nanmean([values12, -values01], axis=0)
            final_mesh[t1] = list(values)
        return final_mesh

    # Wrappers for parallel processing

    def _calc_local_density_mesh(self, item: int):
        timepoint = item
        print('Calculating average density {}'.format(timepoint))
        return timepoint, self.calc_average_tri_density(timepoint)

    def _calc_delta_divergence_mesh(self, item: int):
        timepoint1 = item
        timepoint2 = timepoint1 + 1
        print('Calculating average divergence {} to {}'.format(timepoint1, timepoint2))
        return timepoint1, timepoint2, self.calc_average_tri_divergence(timepoint1, timepoint2)

    def _calc_delta_curl_mesh(self, item: int):
        timepoint1 = item
        timepoint2 = timepoint1 + 1
        print('Calculating average curl {} to {}'.format(timepoint1, timepoint2))
        return timepoint1, timepoint2, self.calc_average_segment_angle(timepoint1, timepoint2)

    def _triangulate_timepoint(self, item: Tuple[int, float]):
        # Parallel triangulation
        timepoint, max_distance = item
        print('Gridding {}'.format(timepoint))
        points = np.array(self.timepoint_coords[timepoint])
        links, tris, perimeters = calc_delaunay_adjacency(points, max_distance=max_distance)
        return timepoint, links, tris, perimeters

    def _warp_timepoint_to_circle(self, timepoint: int):
        # Parallel warping of the timepoints
        print('Warping {}'.format(timepoint))
        points = np.array(self.timepoint_coords[timepoint])
        best_perimeter = self.get_longest_perimeter(timepoint)
        if best_perimeter is None:
            print('Got invalid perimeter for timepoint {}'.format(timepoint))
            return timepoint, None
        return timepoint, warp_to_circle(points, best_perimeter, i_max=self.radial_samples, r_max=1.1)

    def _calc_track_persistence(self, trackidx):
        print('Analyzing track #{}'.format(trackidx))
        # Parallel calculation of the track persistence stats
        tidx, xidx, rxx, ryy, rtt, x_rad, y_rad = self.get_track_values(
            trackidx, 'timepoint_real_coords', 'timepoint_warp_coords')

        r_rad = np.sqrt(x_rad**2 + y_rad**2)
        tt, xx, yy = np.array(rtt/self.time_scale), np.array(rxx), np.array(ryy)
        persistence = calc_track_persistence(tt, xx, yy,
                                             time_scale=self.time_scale,
                                             space_scale=1.0,  # Already in real coordinates
                                             resample_factor=self.resample_factor,
                                             smooth_points=self.smooth_points,
                                             interp_points=self.interp_points,
                                             min_persistence_points=self.min_persistence_points)
        if persistence is None:
            return trackidx, hdf5_cache.PersistenceStats(
                pct_quiescent=np.nan,
                pct_persistent=np.nan,
                r_rad=r_rad[0],
                x_rad=x_rad[0],
                y_rad=y_rad[0],
                x_pos=xx[0],
                y_pos=yy[0],
                disp=np.nan,
                dist=np.nan,
                disp_to_dist=np.nan,
                vel=np.nan,
                tt=rtt,
                xx=rxx,
                yy=ryy,
                mask=np.full_like(rtt, fill_value=np.nan),
                waveform=np.full_like(rtt, fill_value=np.nan),
                tidx=tidx,
                xidx=xidx,
            )

        # FIXME: I think this can be replaced with ``persistence.average_velocity``
        # Calculate the deltas and final track values
        duration = (rtt[-1] - rtt[0])
        vel = persistence.displacement / duration

        return trackidx, hdf5_cache.PersistenceStats(
            pct_quiescent=persistence.pct_quiescent,
            pct_persistent=persistence.pct_persistent,
            r_rad=r_rad[0],
            x_rad=x_rad[0],
            y_rad=y_rad[0],
            x_pos=xx[0],
            y_pos=yy[0],
            disp=persistence.displacement,
            dist=persistence.distance,
            disp_to_dist=persistence.disp_to_dist,
            vel=vel,
            tt=persistence.sm_tt*self.time_scale,
            xx=persistence.sm_xx,
            yy=persistence.sm_yy,
            mask=persistence.mask,
            waveform=persistence.waveform,
            tidx=tidx,
            xidx=xidx,
        )

    # Per-timepoint calculations

    def calc_average_tri_density(self, timepoint: int):
        """ Average density using triangles

        :param int timepoint:
            The timepoint to load
        :returns:
            A density value at each point or None
        """
        if timepoint not in self.timepoint_triangles or timepoint not in self.timepoint_real_coords:
            return None

        # N x 2 array of points in um space
        timepoint_points = np.array(self.timepoint_real_coords[timepoint])
        # Have to convert from a set of tuples to a list of tuples
        timepoint_triangles = np.array(list(self.timepoint_triangles[timepoint]))

        if timepoint_points.shape[0] < 1 or timepoint_triangles.shape[0] < 1:
            return None

        assert timepoint_points.shape[1] == 3
        assert timepoint_triangles.shape[1] == 3

        return calc_average_tri_density(timepoint_points[:, :2].astype(np.float64),
                                        timepoint_triangles.astype(np.int64),
                                        timepoint_points.shape[0],
                                        timepoint_triangles.shape[0])

    def calc_average_tri_divergence(self, timepoint1: int, timepoint2: int):
        """ Divergence between timepoints

        :param int timepoint1:
            The earlier timepoint to load
        :param int timepoint2:
            The later timepoint to load
        :returns:
            Forward (density2 - density1) and reverse (density1 - density2) divergences for each point
        """

        if timepoint1 not in self.local_densities_mesh or timepoint2 not in self.local_densities_mesh:
            return None, None
        if (timepoint1, timepoint2) not in self.timepoint_links:
            return None, None
        density1 = np.array(self.local_cell_areas_mesh[timepoint1])
        density2 = np.array(self.local_cell_areas_mesh[timepoint2])

        timepoint_links = np.array(list(self.timepoint_links[timepoint1, timepoint2].items()))

        return calc_average_segment_divergence(density1.astype(np.float64),
                                               density2.astype(np.float64),
                                               timepoint_links.astype(np.int64),
                                               density1.shape[0],
                                               density2.shape[0],
                                               timepoint_links.shape[0],
                                               float(self.time_scale))

    def calc_average_segment_angle(self, timepoint1: int, timepoint2: int):
        """ Average curl using meshes

        :param int timepoint1:
            The early timepoint to load
        :param int timepoint2:
            The late timepoint to load
        :returns:
            The forward angle change (rot2 - rot1) and the reverse angle change (rot1 - rot2)
        """
        if (timepoint1, timepoint2) not in self.timepoint_links:
            return None, None
        if timepoint1 not in self.timepoint_meshes:
            return None, None
        if timepoint1 not in self.timepoint_real_coords:
            return None, None
        if timepoint2 not in self.timepoint_real_coords:
            return None, None

        timepoint1_points = np.array(self.timepoint_real_coords[timepoint1])
        timepoint2_points = np.array(self.timepoint_real_coords[timepoint2])

        assert timepoint1_points.shape[1] == 3
        assert timepoint2_points.shape[1] == 3

        timepoint_links = self.timepoint_links[timepoint1, timepoint2]
        timepoint_mesh = self.timepoint_meshes[timepoint1]

        return calc_average_segment_angle(timepoint1_points[:, :2].astype(np.float64),
                                          timepoint2_points[:, :2].astype(np.float64),
                                          timepoint_mesh,
                                          timepoint_links,
                                          timepoint1_points.shape[0],
                                          timepoint2_points.shape[0])

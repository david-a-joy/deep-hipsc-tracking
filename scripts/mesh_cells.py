#!/usr/bin/env python3
"""  Mesh detected cells to produce colony segmentations

Triangulate an experiment:

.. code-block:: bash

    $ ./mesh_cells.py -r /path/to/experiment

Generate the plots for the paper:

.. code-block:: bash

    $ ./mesh_cells.py \\
        --plot-style light \\
        --suffix '.svg' \\
        --detector Composite \\
        -r /data/Experiment/2017-08-29/

"""

# Standard lib
import sys
import shutil
import pathlib
import argparse
import traceback
from typing import Tuple, List, Optional

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party imports
import numpy as np

from scipy.ndimage.filters import gaussian_filter1d

import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Our own imports
from deep_hipsc_tracking.presets import load_preset
from deep_hipsc_tracking.stats import GridDB
from deep_hipsc_tracking.plotting import (
    set_plot_style, add_violins_with_outliers, add_gradient_line, add_meshplot,
    add_poly_meshplot,
)
from deep_hipsc_tracking.tracking import load_track_csvfile
from deep_hipsc_tracking.utils import (
    load_image, Hypermap, area_of_polygon,
    calc_pairwise_significance, bin_by_radius
)

# Constants
PLOT_INDIVIDUAL_MESHES = False  # If True, plot a mesh image for each timepoint
TIMEPOINT_STEP = -1  # Frequency to generate timepoints, or -1 for none

FIGSIZE = (24, 24)  # Size for the figures
VIOLIN_FIGSIZE = (8, 8)  # Size for the violinplots

PROCESSES = 12  # Number of parallel processes to use

MAX_BIN_RADIUS = 0.99  # maximum radius to pull binned curl/divergence

PLOT_STYLE = 'light'
SUFFIX = '.png'

# y-limits for categories of the form ymin, ymax
CATEGORY_LIMITS = {
    'cell_area': (40, 120),  # um^2 - Area of the segmented cell
    'curl': (-0.3, 0.3),  # rads/min - How much a cell rotates each frame
    'density': (0.0, 0.03),  # um^-2 - Cell density
    'displacement': (0, 50),  # um - How much the cells move in 6 hours
    'distance': (0, 120),  # um - Long the cell track is after 6 hours
    'divergence': (-4, 4),  # um^2/min - How much the cells spread over 6 hours
    'disp_to_dist': (-0.1, 1.1),  # ratio - 0 is stopped, 1 moves perfectly straight
    'persistence': (-0.1, 1.1),  # ratio - how much of the cell track is movement
    'velocity': (0.0, 0.5),  # um/min - how fast does the cell move
}
CATEGORY_LABELS = {
    'cell_area': 'Cell Area ($\\mu m^2$)',
    'density': 'Cell Density ($\\mu m^{-2}$)',
    'divergence': 'Area Fold Change',
    'curl': 'Cell Rotation (radians/min)',
    'persistence': 'Percent Persistent Migration',
    'velocity': 'Velocity ($\\mu m / min$)',
    'distance': 'Distance Traveled ($\\mu m$)',
    'displacement': 'Cell Displacement ($\\mu m$)',
    'disp_to_dist': 'Displacement vs Distance',
}


# Classes


class TriangulatedStatsPlotter(object):
    """ Plot the stats for a grid database over a single colony

    :param GridDB grid:
        The grid data to render
    :param Path imagedir:
        Path to write the stat plots out to
    :param int tile:
        The tile number for these stats
    :param tuple[float] figsize:
        Figure size for the main plots
    :param tuple[float] violin_figsize:
        Figure size for the violin plots
    :param float max_bin_radius:
        Maximum radius to bin the radial stats over (0 is the center 1 is the perimeter)
    :param int processes:
        Number of parallel processes to use while replotting
    """

    def __init__(self,
                 grid: GridDB,
                 imagedir: pathlib.Path,
                 tile: str,
                 outdir: Optional[pathlib.Path] = None,
                 plot_style: str = PLOT_STYLE,
                 suffix: str = SUFFIX,
                 figsize: Tuple[float] = FIGSIZE,
                 violin_figsize: Tuple[float] = VIOLIN_FIGSIZE,
                 max_bin_radius: float = MAX_BIN_RADIUS,
                 processes: int = PROCESSES):

        # Database object
        self.grid = grid

        # Image directory finding
        self.imagedir = imagedir
        self.tile = tile

        # Plot directory
        self.processes = processes
        self.outdir = outdir
        self.tile_outdir = None

        # Plot style controls
        self.plot_style = plot_style
        self.suffix = suffix
        self.figsize = figsize
        self.violin_figsize = violin_figsize

        # Limits for plots
        self.divergence_cmap = 'coolwarm'
        self.divergence_min = CATEGORY_LIMITS['divergence'][0]
        self.divergence_max = CATEGORY_LIMITS['divergence'][1]

        self.curl_cmap = 'coolwarm'
        self.curl_min = CATEGORY_LIMITS['curl'][0]
        self.curl_max = CATEGORY_LIMITS['curl'][1]

        self.cell_area_cmap = 'inferno'
        self.cell_area_min = CATEGORY_LIMITS['cell_area'][0]
        self.cell_area_max = CATEGORY_LIMITS['cell_area'][1]

        self.density_cmap = 'inferno'
        self.density_min = CATEGORY_LIMITS['density'][0]
        self.density_max = CATEGORY_LIMITS['density'][1]

        self.velocity_cmap = 'inferno'
        self.velocity_min = CATEGORY_LIMITS['velocity'][0]
        self.velocity_max = CATEGORY_LIMITS['velocity'][1]

        self.persistence_cmap = 'inferno'
        self.persistence_min = CATEGORY_LIMITS['persistence'][0]
        self.persistence_max = CATEGORY_LIMITS['persistence'][1]

        self.distance_cmap = 'inferno'
        self.distance_min = CATEGORY_LIMITS['distance'][0]
        self.distance_max = CATEGORY_LIMITS['distance'][1]

        self.displacement_cmap = 'inferno'
        self.displacement_min = CATEGORY_LIMITS['displacement'][0]
        self.displacement_max = CATEGORY_LIMITS['displacement'][1]

        self.disp_to_dist_cmap = 'inferno'
        self.disp_to_dist_min = CATEGORY_LIMITS['disp_to_dist'][0]
        self.disp_to_dist_max = CATEGORY_LIMITS['disp_to_dist'][1]

        # Index information for times
        self.all_timepoints = []
        self.key_timepoints = []

        # Smoothing for timeseries
        self.area_smoothing = 1.0  # Sigma to smooth area changes

        self.timepoint_image_shapes = {}

        # Containers for field values
        self.mean_density = None
        self.mean_divergence = None
        self.mean_curl = None

        self.mean_warp_density = None
        self.mean_warp_divergence = None
        self.mean_warp_curl = None

        self.perimeters = []
        self.areas = []
        self.smoothed_areas = None
        self.delta_areas = None

        # Image file information
        self.imagefile = None
        self.image = None
        self.rows, self.cols = None, None
        self.warp_rows, self.warp_cols = grid.radial_samples, grid.radial_samples

        # Radially warped field data
        self.grid_radius = None
        self.grid_mean_density = None
        self.grid_mean_divergence = None
        self.grid_mean_curl = None

        # Radially warped track data
        self.max_bin_radius = max_bin_radius
        self.track_radius = None
        self.track_mean_velocity = None
        self.track_mean_distance = None
        self.track_mean_displacement = None
        self.track_mean_persistence = None

    def make_plot_outdir(self):
        """ Make the plot directory """

        if self.outdir is not None:
            tile_outdir = self.outdir / self.tile
            if tile_outdir.is_dir():
                print(f'Clearing old plots: {tile_outdir}')
                shutil.rmtree(str(tile_outdir))
            tile_outdir.mkdir(parents=True, exist_ok=True)
        else:
            tile_outdir = None
        self.tile_outdir = tile_outdir

    def load_reference_image(self):
        """ Load the reference image """

        print('Plotting...')

        self.imagefile = find_timepoint(
            self.imagedir, tile=self.tile, timepoint=min(self.grid.timepoint_coords.keys()))
        self.image = load_image(self.imagefile)
        self.rows, self.cols = self.image.shape[:2]

    def load_timepoints(self):
        """ Load the timepoints we want """

        self.all_timepoints = self.grid.get_timepoint_range()
        self.key_timepoints = []

        if self.all_timepoints:
            min_timepoint = min(self.all_timepoints)
            max_timepoint = max(self.all_timepoints)

            mean_timepoint = int(round((max_timepoint + min_timepoint)/2))

            self.key_timepoints.append(min_timepoint)
            if mean_timepoint > min_timepoint:
                self.key_timepoints.append(mean_timepoint)

            if max_timepoint-1 > min_timepoint:
                self.key_timepoints.append(max_timepoint-1)
            elif max_timepoint > min_timepoint:
                self.key_timepoints.append(max_timepoint)

    def load_perimeters(self):
        """ Load the perimeters for each timepoint """

        perimeters = []
        areas = []

        for timepoint in self.all_timepoints:
            try:
                perimeter = self.grid.get_longest_perimeter(timepoint)
            except KeyError:
                continue

            perimeter = np.concatenate([perimeter, perimeter[0:1, :]], axis=0)
            perimeters.append(perimeter)

            areas.append(area_of_polygon(perimeter))

        self.perimeters = np.array(perimeters)
        self.areas = np.array(areas)

        # Smooth the timeseries and calculate dA/A0
        if self.areas.shape[0] > 0:
            self.smoothed_areas = gaussian_filter1d(self.areas, self.area_smoothing)
            self.delta_areas = (self.smoothed_areas[1:] - self.smoothed_areas[:-1])/self.smoothed_areas[0]
        else:
            self.smoothed_areas = np.array([])
            self.delta_areas = np.array([])

    def load_coord_mesh(self, fieldname: str, timepoint: int) -> np.ndarray:
        """ Load the coordinates for this timepoint

        :param str fieldname:
            The field to load from the database
        :param int timepoint:
            The timepoint to load from the database
        :returns:
            The mesh as a numpy array
        """

        if fieldname == 'image':
            return np.array(self.grid.timepoint_coords[timepoint])
        elif fieldname == 'warp':
            return np.array(self.grid.timepoint_warp_coords[timepoint])
        elif fieldname == 'real':
            x, y, _ = self.grid.timepoint_real_coords[timepoint]
            return np.array([x, y])
        elif fieldname == 'mesh':
            return self.grid.timepoint_meshes[timepoint]
        elif fieldname in ('tris', 'triangles'):
            return self.grid.timepoint_triangles[timepoint]
        else:
            raise KeyError(f'Unknown coordinate type: "{fieldname}"')

    def load_field_mesh(self, fieldname: str, timepoint: int) -> np.ndarray:
        """ Composite fields using the new mesh (pointwise) system

        :param str fieldname:
            The field to composite
        :param int timepoint:
            The timepoint to load
        :returns:
            The values at that timepoint
        """
        print(f'Loading "{fieldname}"')
        attr = {
            'density': 'local_densities_mesh',
            'cell_area': 'local_cell_areas_mesh',
            'divergence': 'delta_divergence_mesh',
            'curl': 'delta_curl_mesh',
            'velocity': 'local_velocity_mesh',
            'speed': 'local_speed_mesh',
            'distance': 'local_distance_mesh',
            'displacement': 'local_displacement_mesh',
            'persistence': 'local_persistence_mesh',
            'disp_to_dist': 'local_disp_vs_dist_mesh',
        }[fieldname]
        return np.array(getattr(self.grid, attr)[timepoint])

    def warp_parameters_mesh(self):
        """ Warp a set of parameters onto the radial coordinate frame """

        self.track_radius = self.grid.get_all_track_summaries('timepoint_warp_radius', func='mean')[0]
        track_time = self.grid.get_all_track_summaries('timepoint_real_coords', func='max')[2]

        self.track_mean_distance = self.grid.get_all_track_summaries('local_distance_mesh', func='max')[0]
        self.track_mean_displacement = self.grid.get_all_track_summaries('local_displacement_mesh', func='max')[0]

        self.track_mean_velocity = self.track_mean_displacement / track_time
        self.track_mean_speed = self.track_mean_distance / track_time

        self.track_mean_persistence = self.grid.get_all_track_summaries('local_persistence_mesh', func='mean')[0]
        self.track_mean_disp_to_dist = self.track_mean_displacement / self.track_mean_distance

        self.track_mean_density = self.grid.get_all_track_summaries('local_densities_mesh', func='mean')[0]
        self.track_mean_cell_area = self.grid.get_all_track_summaries('local_cell_areas_mesh', func='mean')[0]
        self.track_mean_divergence = self.grid.get_all_track_summaries('delta_divergence_mesh', func='mean')[0]
        self.track_mean_curl = self.grid.get_all_track_summaries('delta_curl_mesh', func='mean')[0]

    # Plots

    def plot_all_single_timepoints(self,
                                   all_fieldnames: List[str],
                                   timepoint_step: int = TIMEPOINT_STEP):
        """ Make all the individual timepoints

        :param str all_fieldnames:
            The list of fieldnames to plot
        :param int timepoint_step:
            Step to take when plotting individual meshes
        """
        items = [(timepoint, all_fieldnames, timepoint_step)
                 for timepoint in self.all_timepoints]
        if self.processes < 1:
            processes = PROCESSES
        else:
            processes = self.processes

        with Hypermap(processes=processes, lazy=True) as pool:
            res = pool.map(self.plot_single_timepoint, items)
        print(f'Plotted {sum(list(res))} timepoints successfully')

    def plot_single_timepoint(self, item: Tuple):
        """ Plot a single timepoint inside a map call

        :param tuple item:
            The data to plot
        :returns:
            True if the plotting worked, False otherwise
        """
        timepoint, all_fieldnames, timepoint_step = item

        # See if we should plot an individual image
        should_plot_individual_mesh = any([
            PLOT_INDIVIDUAL_MESHES,
            timepoint_step > 0 and (timepoint % timepoint_step == 0),
            timepoint in self.key_timepoints,
        ])
        if not should_plot_individual_mesh:
            return False

        if timepoint not in self.grid.timepoint_coords:
            return False
        if timepoint not in self.grid.timepoint_meshes:
            return False

        try:
            self.plot_single_timepoint_mesh(timepoint)
            for fieldname in all_fieldnames:
                self.plot_single_timepoint_mesh_field(fieldname, timepoint)
        except Exception:
            traceback.print_exc()
            return False
        return True

    def plot_single_timepoint_mesh(self, timepoint: int):
        """ Plot the mesh at a single timepoint

        :param int timepoint:
            Timepoint to plot the mesh at
        """

        # Load the image to plot over
        imagefile = find_timepoint(self.imagedir, tile=self.tile, timepoint=timepoint)
        image = load_image(imagefile)

        # Load the mesh to plot
        points = self.load_coord_mesh('image', timepoint)
        warp_points = self.load_coord_mesh('warp', timepoint)
        mesh = self.load_coord_mesh('mesh', timepoint)

        if len(self.perimeters) > timepoint:
            perimeter = self.perimeters[timepoint]
        else:
            perimeter = None

        rows, cols = image.shape[:2]
        self.timepoint_image_shapes[timepoint] = (rows, cols)

        # Plot the triangulation over the original image
        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'Mesh-{self.tile}t{timepoint:03d}{self.suffix}'
            outfile = self.tile_outdir / 'Mesh' / outfile
            outfile.parent.mkdir(parents=True, exist_ok=True)
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.imshow(image, cmap='bone')
            add_meshplot(ax, points, mesh)

            ax.set_xlim([0, cols])
            ax.set_ylim([rows, 0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_axis_off()

            style.show(outfile, tight_layout=True)

        # Plot the current perimeter
        if perimeter is not None:
            if self.tile_outdir is None:
                outfile = None
            else:
                outfile = f'Perimeter-{self.tile}t{timepoint:03d}{self.suffix}'
                outfile = self.tile_outdir / 'Perimeter' / outfile
                outfile.parent.mkdir(parents=True, exist_ok=True)
            with set_plot_style(self.plot_style) as style:
                fig, ax = plt.subplots(1, 1, figsize=self.figsize)
                ax.plot(perimeter[:, 0], perimeter[:, 1], '-r')
                ax.imshow(image, cmap='bone')

                ax.set_xlim([0, cols])
                ax.set_ylim([rows, 0])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
                ax.set_axis_off()

                style.show(outfile, tight_layout=True)

        # Plot the warped mesh
        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'Warp-{self.tile}t{timepoint:03d}{self.suffix}'
            outfile = self.tile_outdir / 'Warp' / outfile
            outfile.parent.mkdir(parents=True, exist_ok=True)
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            add_meshplot(ax, warp_points, mesh)

            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_axis_off()

            style.show(outfile, tight_layout=True)

    def plot_single_timepoint_mesh_field(self, fieldname: str, timepoint: int):
        """ Plot the field and mesh from a single timepoint

        :param str fieldname:
            The field to load
        :param int timepoint:
            The timepoint to load
        """
        coords = self.load_coord_mesh('image', timepoint)
        warp_coords = self.load_coord_mesh('warp', timepoint)
        tris = self.load_coord_mesh('tris', timepoint)

        field = self.load_field_mesh(fieldname, timepoint)

        cmap = getattr(self, f'{fieldname}_cmap')
        vmin = getattr(self, f'{fieldname}_min')
        vmax = getattr(self, f'{fieldname}_max')

        if fieldname in ('distance', 'displacement'):
            max_timepoint = max(self.all_timepoints)
            vmin = vmin * (timepoint / max_timepoint)
            vmax = vmax * (timepoint / max_timepoint)

        rows, cols = self.timepoint_image_shapes.get(timepoint,
                                                     (np.max(coords[:, 0]), np.max(coords[:, 1])))

        name = fieldname.capitalize()

        # Plot the triangulation over the original image
        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'{name}-{self.tile}t{timepoint:03d}{self.suffix}'
            outfile = self.tile_outdir / name / outfile
            outfile.parent.mkdir(exist_ok=True, parents=True)
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            add_poly_meshplot(ax, coords, tris, field, vmin=vmin, vmax=vmax, cmap=cmap)

            ax.set_xlim([0, cols])
            ax.set_ylim([rows, 0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_axis_off()

            style.show(outfile, tight_layout=True)

        # Plot the warped mesh
        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'Warp{name}-{self.tile}t{timepoint:03d}{self.suffix}'
            outfile = self.tile_outdir / f'Warp{name}' / outfile
            outfile.parent.mkdir(exist_ok=True, parents=True)
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            add_poly_meshplot(ax, warp_coords, tris, field, vmin=vmin, vmax=vmax, cmap=cmap)

            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_axis_off()

            style.show(outfile, tight_layout=True)

    def plot_perimeter_timeseries(self):
        """ Plot the perimeters on one plot """

        # FIXME: Should we smooth them first?

        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'Perimeters-{self.tile}{self.suffix}'
            outfile = self.tile_outdir / outfile
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            for perimeter in self.perimeters:
                ax.plot(perimeter[:, 0], perimeter[:, 1])
            ax.set_xticks([])
            ax.set_yticks([])

            style.show(outfile, tight_layout=True)

        # FIXME: Look at perimeter change, dP/P0 here...

    def plot_area_timeseries(self):
        """ Plot all the area changes over time """

        # Plot the areas over time
        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'Areas-{self.tile}{self.suffix}'
            outfile = self.tile_outdir / outfile
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.plot(np.arange(len(self.areas)), self.areas)
            ax.set_title('Colony area over time')
            ax.set_xlabel('Frame #')
            ax.set_ylabel('Colony area')

            style.show(outfile, tight_layout=True)

        # Plot the delta area over time
        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'DeltaAreas-{self.tile}{self.suffix}'
            outfile = self.tile_outdir / outfile
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.plot(np.arange(len(self.delta_areas)), self.delta_areas)
            ax.set_title('Change in colony area (dA/$A_0$)')
            ax.set_xlabel('Frame #')
            ax.set_ylabel('Delta colony area (dA/$A_0$)')

            style.show(outfile, tight_layout=True)

    def plot_field_mesh(self, fieldname: str):
        """ Plot the average values of the field over all time """
        print('No field mesh plots yet...')

    def plot_field(self, fieldname: str):
        """ Plot all the properties for the field

        :param str fieldname:
            The name of the field to plot
        """
        mean_field = getattr(self, f'mean_{fieldname}')
        mean_warp_field = getattr(self, f'mean_warp_{fieldname}')

        cmap = getattr(self, f'{fieldname}_cmap')
        vmin = getattr(self, f'mean_{fieldname}_min')
        vmax = getattr(self, f'mean_{fieldname}_max')

        # Plot the unwarped field
        if mean_field is None:
            print(f'No mean field data for {fieldname}')
        else:
            if self.tile_outdir is None:
                outfile = None
            else:
                outfile = f'{fieldname.capitalize()}-{self.tile}{self.suffix}'
                outfile = self.tile_outdir / outfile
            with set_plot_style(self.plot_style) as style:
                fig, ax = plt.subplots(1, 1, figsize=self.figsize)
                ax.imshow(mean_field, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])

                style.show(outfile, tight_layout=True)

        # Plot the warped field
        if mean_warp_field is None:
            print(f'No mean warp field data for {fieldname}')
        else:
            if self.tile_outdir is None:
                outfile = None
            else:
                outfile = f'Warp{fieldname.capitalize()}-{self.tile}{self.suffix}'
                outfile = self.tile_outdir / outfile
            with set_plot_style(self.plot_style) as style:
                fig, ax = plt.subplots(1, 1, figsize=self.figsize)
                ax.imshow(mean_warp_field, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])

                style.show(outfile, tight_layout=True)

    def plot_persistence_timeseries(self):
        """ Plot the persistent tracks on one plot """

        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = self.tile_outdir / f'Tracks-{self.tile}{self.suffix}'
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.plot(self.perimeters[-1][:, 0],
                    self.perimeters[-1][:, 1], '-r', linewidth=2)

            for track in self.grid.track_peristences.values():
                if track is None:
                    continue
                add_gradient_line(ax,
                                  track.xx/self.grid.space_scale,
                                  track.yy/self.grid.space_scale,
                                  track.mask,
                                  vmin=-0.1, vmax=1.1, cmap='Dark2')
            ax.autoscale_view()

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_axis_off()

            style.show(outfile, tight_layout=True)

        # FIXME: Plot average persistence in space

        # FIXME: Look into bundling persistence over tracks

    def plot_parameter_violins(self,
                               parameter_name: str,
                               num_bins: int,
                               bin_type: str = 'area',
                               graphic: str = 'violin'):
        """ Plot violin distributions for different parameters

        :param str parameter_name:
            Name of the parameter to plot
        :param int num_bins:
            Number of radial bins to divide the data into
        :param str bin_type:
            How to equally divide the bins (one of 'radius' or 'area')
        :param str graphic:
            Which graphic to plot the parameters on ('violin', 'bins', 'boxes')
        """
        if parameter_name in ('divergence', 'curl'):
            extremes = 'both'
        else:
            extremes = 'upper'

        radius_attr = 'track_radius'
        parameter_attr = f'track_mean_{parameter_name}'

        radius_data = getattr(self, radius_attr, None)
        parameter_data = getattr(self, parameter_attr, None)

        if radius_data is None or parameter_data is None:
            print(f'No radial data for {parameter_attr} vs {radius_attr}')
            return None

        ymin, ymax = CATEGORY_LIMITS.get(parameter_name, (None, None))
        ylabel = CATEGORY_LABELS.get(parameter_name, None)

        print(f'Plotting {parameter_attr} vs {radius_attr}')

        # Bin the gridded density
        data = bin_by_radius(radius_data, parameter_data,
                             num_bins=num_bins, bin_type=bin_type,
                             label=parameter_name.capitalize())
        # Calculate the significance
        significance = calc_pairwise_significance(data,
                                                  category='Radius',
                                                  score=parameter_name.capitalize())
        ycolumn = parameter_name.capitalize()
        if self.tile_outdir is None:
            outfile = None
        else:
            outfile = f'{ycolumn}VsRadius-{graphic}-{self.tile}-{num_bins:d}bins{self.suffix}'
            outfile = self.tile_outdir / outfile
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.violin_figsize)
            add_violins_with_outliers(ax, data,
                                      xcolumn='Radius',
                                      ycolumn=ycolumn,
                                      extremes=extremes,
                                      significance=significance,
                                      savefile=outfile,
                                      graphic=graphic,
                                      ymin=ymin,
                                      ymax=ymax,
                                      ylabel=ylabel)
            style.show(outfile, tight_layout=True)

        return significance

# Functions


def find_timepoint(imagedir: pathlib.Path,
                   tile: str,
                   timepoint: int = 1) -> pathlib.Path:
    """ Find the specified timepoint

    :param Path imagedir:
        Directory containing the image file
    :param str tile:
        Prefix for all images in this directory
    :param int timepoint:
        Time index for which frame to load (1 - first frame, 2 - second, etc)
    :returns:
        The file matching this timestamp
    """
    ct = 0
    for infile in sorted(imagedir.iterdir()):
        if not infile.is_file():
            continue
        if infile.suffix not in ('.png', '.tif', '.jpg'):
            continue
        if not infile.name.startswith(tile):
            continue
        ct += 1
        if ct == timepoint:
            return infile
    raise ValueError(f'No images under {imagedir} match tile {tile} and timepoint {timepoint}')


def plot_triangulated_stats(grid: GridDB,
                            imagedir: pathlib.Path,
                            outdir: Optional[pathlib.Path] = None,
                            skip_single_timepoints: bool = False,
                            **kwargs):
    """ Plot the stats for a single track

    :param GridDB grid:
        The grid data to render
    :param Path imagedir:
        Path to the image files to load
    :param Path outdir:
        If not None, the path to write the plots out to
    :param \\*\\* kwargs:
        Arguments to pass to :py:class:`TriangulatedStatsPlotter`
    """
    tile = imagedir.name

    all_fieldnames = ['density', 'cell_area', 'divergence', 'curl',
                      'velocity', 'distance', 'displacement', 'persistence',
                      'disp_to_dist']
    all_radial_bins = [3]

    plotter = TriangulatedStatsPlotter(
        grid, imagedir, tile, outdir=outdir, **kwargs)
    plotter.make_plot_outdir()
    plotter.load_reference_image()
    plotter.load_timepoints()

    # Load in the time-dependent items
    plotter.load_perimeters()

    # Warp the parameters onto a radial coordinate system
    plotter.warp_parameters_mesh()

    # Make single timepoint plots for debugging
    if not skip_single_timepoints:
        plotter.plot_all_single_timepoints(all_fieldnames)

    # Composite plots
    plotter.plot_perimeter_timeseries()
    plotter.plot_area_timeseries()
    plotter.plot_persistence_timeseries()

    for fieldname in all_fieldnames:
        plotter.plot_field_mesh(fieldname)

    all_significance = {
        'Name': [],
        'Bin1': [],
        'Bin2': [],
        'Pvalue': [],
    }
    for num_bins in all_radial_bins:
        for parameter_name in all_fieldnames:
            significance = None
            for graphic in ['violin', 'box', 'bar']:
                sig = plotter.plot_parameter_violins(parameter_name, num_bins, graphic=graphic)
                if significance is None:
                    significance = sig
            if significance is None:
                continue
            for (key1, key2), pvalue in significance.items():
                all_significance['Name'].append(parameter_name)
                all_significance['Bin1'].append(key1)
                all_significance['Bin2'].append(key2)
                all_significance['Pvalue'].append(pvalue)

    if len(all_significance['Name']) > 0:
        all_significance = pd.DataFrame(all_significance)
        all_significance.to_excel(outdir / f'significance_{tile}.xlsx')


# Main function


def calc_triangulated_stats(rootdir: pathlib.Path,
                            config_file: pathlib.Path,
                            processes: int = PROCESSES,
                            plot_style: str = PLOT_STYLE,
                            suffix: str = SUFFIX,
                            overwrite: bool = False):
    """ Calculate the triangulated stats

    :param Path rootdir:
        The experiment directory to process
    :param int processes:
        How many parallel processes to use
    :param int max_timepoint:
        The maximum timepoint to segment to
    :param str plot_style:
        The stylesheet for the plots
    :param str suffix:
        The suffix to save plots with
    :param float max_distance:
        The maximum distance to connect cells across
    :param str detector:
        The detector to use the triangulation from
    :param bool overwrite:
        If True, overwrite the data cache
    """
    config = load_preset(config_file)
    time_scale = config.time_scale
    space_scale = config.space_scale

    skip_plots = config.meshing['skip_plots']
    skip_single_timepoints = config.meshing['skip_single_timepoints']
    max_distance = config.meshing['max_distance']

    detectors = config.meshing['detectors']
    if detectors in ([], None):
        detector = None
    elif len(detectors) == 1:
        detector = detectors[0]
    else:
        raise ValueError(f'Can only mesh a single tracked detector: got {detectors}')

    if detector is None:
        trackdir_name = 'CellTracking'
        outdir_name = 'GridCellTracking'
    else:
        for try_detector in [detector.lower(), detector.capitalize()]:
            trackdir_name = f'CellTracking-{try_detector}'
            if (rootdir / trackdir_name).is_dir():
                break
            outdir_name = f'GridCellTracking-{try_detector.capitalize()}'

    trackdir = rootdir / trackdir_name / 'Tracks'
    image_rootdir = rootdir / 'Corrected'

    if overwrite:
        if detector is None:
            try_outdirs = ['GridCellTracking']
        else:
            try_outdirs = [f'GridCellTracking-{d}'
                           for d in (detector.lower(), detector.capitalize())]
        for try_outdir in try_outdirs:
            if (rootdir / try_outdir).is_dir():
                print(f'Overwriting: {rootdir / try_outdir}')
                shutil.rmtree(str(rootdir / try_outdir))
    outdir = rootdir / outdir_name

    grid_outdir = outdir / 'gridded_tiles'
    plot_outdir = outdir / 'plots'

    if not grid_outdir.is_dir():
        grid_outdir.mkdir(parents=True)
    if not plot_outdir.is_dir():
        plot_outdir.mkdir(parents=True)

    for trackfile in sorted(trackdir.iterdir()):
        if not trackfile.name.endswith('_traces.csv'):
            continue
        if not trackfile.is_file():
            continue
        print(f'Loading tracks from {trackfile}')
        trackstem = trackfile.name[:-len('_traces.csv')]

        grid_outfile = grid_outdir / f'gridded_{trackstem}.h5'
        if grid_outfile.is_file():
            if skip_plots:
                print(f'Skipping cached file: {grid_outfile}')
                grid = None
            else:
                print(f'Loading cached file: {grid_outfile}')
                grid = GridDB.from_hdf5(grid_outfile)
                grid.processes = processes
        else:
            grid = GridDB(processes=processes,
                          time_scale=time_scale,
                          space_scale=space_scale)
            tracks = load_track_csvfile(trackfile)
            for track in tracks:
                grid.add_track(track)
            grid.triangulate_grid(max_distance=max_distance)
            grid.warp_grid_to_circle()
            grid.calc_radial_stats()
            grid.calc_local_densities_mesh()
            grid.calc_delta_divergence_mesh()
            grid.calc_delta_curl_mesh()

            print(f'Saving cache to: {grid_outfile}')
            grid.to_hdf5(grid_outfile)

        # Make plots based on the grid database we built
        if skip_plots:
            print(f'Skipping plots for {trackfile}')
        else:
            imagedir = image_rootdir / trackstem
            print(f'Generating plots for {trackfile}')
            plot_triangulated_stats(grid, imagedir,
                                    processes=processes,
                                    outdir=plot_outdir,
                                    plot_style=plot_style,
                                    suffix=suffix,
                                    skip_single_timepoints=skip_single_timepoints)


# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rootdir', dest='rootdirs',
                        action='append', default=[],
                        type=pathlib.Path,
                        help='Root directory to process')
    parser.add_argument('--config-file', type=pathlib.Path,
                        help='Path to the global configuration file')
    parser.add_argument('--processes', type=int, default=Hypermap.cpu_count(),
                        help='Number of parallel processes to use')
    parser.add_argument('--plot-style', default=PLOT_STYLE,
                        help='Style for the plots')
    parser.add_argument('--suffix', default=SUFFIX,
                        help='Suffix for the plot files')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    rootdirs = args.pop('rootdirs')
    num_errors = 0
    for rootdir in rootdirs:
        try:
            calc_triangulated_stats(rootdir=rootdir, **args)
        except Exception:
            print(f'Error processing {rootdir}')
            traceback.print_exc()
            num_errors += 1
    if num_errors > 0:
        raise RuntimeError(f'Got {num_errors} errors during processing')


if __name__ == '__main__':
    main()

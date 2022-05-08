#!/usr/bin/env python3

""" Interface to the HDF5 database

Base class:

* :py:class:`BaseHDF5Cache`: Interface to h5py

Individual datasets:

* :py:class:`HDF5CoordCache`: Store coordinate to track index maps

"""

# Imports
import pathlib
from collections import namedtuple

# 3rd party
import numpy as np

import h5py


# Classes


PersistenceStats = namedtuple('PersistenceStats', [
    'pct_quiescent', 'pct_persistent', 'r_rad', 'x_rad', 'y_rad',
    'x_pos', 'y_pos', 'disp', 'dist', 'vel', 'disp_to_dist',
    'tt', 'xx', 'yy', 'mask', 'waveform',
    'tidx', 'xidx',
])


class LazyLoader(object):
    """ Lazy loader for attributes """

    def __init__(self, db, loader):
        self._db = db
        self._loader = loader

    def load(self):
        return self._loader(self._db)


class BaseHDF5Cache(object):
    """ Cache coordinates to/from an hdf5 file

    Create a loader by subclassing and implementing:

    .. code-block:: python

        class MyLoader(BaseHDF5Cache):

            def _to_hdf5(self, db, data):
                " Implement writing data to an HDF5 db "

            def _from_hdf5(self, db):
                " Implement loading data from an HDF5 db "

    This class handles lazy loading from the object if requested and namespaces
    in the HDF5 object.
    """

    def __init__(self, hdf_attr):
        self.hdf_attr = hdf_attr

    def load(self, infile):
        """ Load the attribute from the file """

        if isinstance(infile, (str, pathlib.Path)):
            db = h5py.File(str(infile), 'a')
            needs_closing = True
        else:
            db = infile
            needs_closing = False

        try:
            res = self.from_hdf5(db)
        finally:
            if needs_closing:
                db.close()
        return res

    def save(self, infile):
        """ Store the attribute to the file """

        if isinstance(infile, (str, pathlib.Path)):
            db = h5py.File(str(infile), 'a')
            needs_closing = True
        else:
            db = infile
            needs_closing = False

        try:
            res = self.to_hdf5(db)
        finally:
            if needs_closing:
                db.close()
        return res

    def from_hdf5(self, db, lazy=False):
        """ Load the attribute from the HDF5 file lazily or immediately """
        if lazy:
            return LazyLoader(db, self._from_hdf5)
        else:
            return self._from_hdf5(db)

    def to_hdf5(self, db, data):
        """ Save the attribute to the HDF5 file immediately """
        # Just to keep the API consistent
        return self._to_hdf5(db, data)

    def _from_hdf5(self, db):
        """ Load the attribute from the HDF5 file """
        raise NotImplementedError('Implement from_hdf5')

    def _to_hdf5(self, db, data):
        """ Write the attribute to the HDF5 file """
        raise NotImplementedError('Implement to_hdf5')


class HDF5CoordCache(BaseHDF5Cache):
    """ Cache coordinate data """

    def _from_hdf5(self, db):
        data = {}
        for key, value in db[self.hdf_attr].items():
            data[int(key[1:])] = [tuple(v) for v in value]
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, value in data.items():
            grp.create_dataset('t{:03d}'.format(key), data=np.array(value))


class HDF5TimepointLinkCache(BaseHDF5Cache):
    """ Cache timepoint-timepoint links """

    def _from_hdf5(self, db):
        data = {}
        for key, value in db[self.hdf_attr].items():
            sidx, eidx = key[1:].split('_', 1)
            data[int(sidx), int(eidx)] = {int(k): int(v) for k, v in value}
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, value in data.items():
            grp.create_dataset('t{:03d}_{:03d}'.format(*key), data=np.array(list(value.items())))


class HDF5TrackLinkCache(BaseHDF5Cache):
    """ Cache for track point to point links """

    def _from_hdf5(self, db):
        data = {}
        for key, value in db[self.hdf_attr].items():
            data[int(key[2:])] = {int(k): int(v) for k, v in value}
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, value in data.items():
            grp.create_dataset('tr{:03d}'.format(key), data=np.array(list(value.items())))


class HDF5TrackLinkInvCache(BaseHDF5Cache):
    """ Cache for track point to point inverse links """

    def _from_hdf5(self, db):
        # FIXME: We probably don't need to load from the database twice, but maybe?
        data = {}
        for key, value in db[self.hdf_attr].items():
            trackidx = int(key[2:])
            for time_idx, point_idx in value:
                point_data = data.setdefault(int(time_idx), {})
                point_data[int(point_idx)] = trackidx
        return data

    def _to_hdf5(self, db, data):
        raise ValueError('HDF5TrackLinkInv should not be saved. Save HDF5TrackLink instead')


class HDF5TimepointMeshCache(BaseHDF5Cache):
    """ Cache for the mesh data """

    def _from_hdf5(self, db):
        data = {}
        for key, value in db[self.hdf_attr].items():
            links = {}
            for l0, l1 in value:
                links.setdefault(int(l0), set()).add(l1)
            data[int(key[1:])] = links
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, value in data.items():
            links = []
            for l0, link in value.items():
                links.extend((l0, l1) for l1 in link)
            grp.create_dataset('m{:03d}'.format(key), data=np.array(links))


class HDF5PersistenceCache(BaseHDF5Cache):
    """ Cache for the Persistence data """

    def __init__(self, *args, **kwargs):
        self.only_summary_stats = kwargs.pop('only_summary_stats', False)
        super(HDF5PersistenceCache, self).__init__(*args, **kwargs)

    def _from_hdf5(self, db):
        data = {}
        for key, subgroup in db[self.hdf_attr].items():
            track_key = int(key[2:])
            # If the old data was None, just load that
            if len(subgroup) == 0:
                data[track_key] = None
                continue
            attrs = {
                'pct_quiescent': float(subgroup.attrs['pct_quiescent']),
                'pct_persistent': float(subgroup.attrs['pct_persistent']),
                'r_rad': float(subgroup.attrs['r_rad']),
                'x_rad': float(subgroup.attrs['x_rad']),
                'y_rad': float(subgroup.attrs['y_rad']),
                'x_pos': float(subgroup.attrs['x_pos']),
                'y_pos': float(subgroup.attrs['y_pos']),
                'disp': float(subgroup.attrs['disp']),
                'dist': float(subgroup.attrs['dist']),
                'disp_to_dist': float(subgroup.attrs.get('disp_to_dist', -1)),
                'vel': float(subgroup.attrs['vel']),
            }
            if self.only_summary_stats:
                attrs.update({k: np.array([])
                              for k in ['tidx', 'xidx', 'tt', 'xx', 'yy', 'mask', 'waveform']})
            else:
                attrs.update({
                    'tidx': [int(t0) for t0 in subgroup['tidx']],
                    'xidx': [int(x0) for x0 in subgroup['xidx']],
                    'tt': np.array(subgroup['tt']),
                    'xx': np.array(subgroup['xx']),
                    'yy': np.array(subgroup['yy']),
                    'mask': np.array(subgroup['mask'], dtype=bool),
                    'waveform': np.array(subgroup['waveform']),
                })
            data[track_key] = PersistenceStats(**attrs)
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, track in data.items():
            # Create a group for each track
            subgroup = grp.create_group('tr{:03d}'.format(key))

            # Leave masked tracks in a blank group
            if track is None:
                continue

            # Store the calculated scalars
            subgroup.attrs['pct_quiescent'] = track.pct_quiescent
            subgroup.attrs['pct_persistent'] = track.pct_persistent
            subgroup.attrs['r_rad'] = track.r_rad
            subgroup.attrs['x_rad'] = track.x_rad
            subgroup.attrs['y_rad'] = track.y_rad
            subgroup.attrs['x_pos'] = track.x_pos
            subgroup.attrs['y_pos'] = track.y_pos
            subgroup.attrs['disp'] = track.disp
            subgroup.attrs['dist'] = track.dist
            subgroup.attrs['disp_to_dist'] = track.disp_to_dist
            subgroup.attrs['vel'] = track.vel

            # Store the actual tracks
            subgroup.create_dataset('tt', data=track.tt)
            subgroup.create_dataset('xx', data=track.xx)
            subgroup.create_dataset('yy', data=track.yy)
            subgroup.create_dataset('mask', data=track.mask)
            subgroup.create_dataset('waveform', data=track.waveform)
            subgroup.create_dataset('tidx', data=np.array(track.tidx, dtype=np.int64))
            subgroup.create_dataset('xidx', data=np.array(track.xidx, dtype=np.int64))


class HDF5CoordValueCache(BaseHDF5Cache):
    """ Cache coordinate data """

    def _from_hdf5(self, db):
        data = {}
        for key, value in db[self.hdf_attr].items():
            data[int(key[1:])] = [float(v) for v in value]
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, value in data.items():
            grp.create_dataset('t{:03d}'.format(key), data=np.array(value))


class HDF5TimepointTriangleCache(BaseHDF5Cache):
    """ Cache for the triangles in each timepoint """

    def _from_hdf5(self, db):
        data = {}
        for key, value in db[self.hdf_attr].items():
            data[int(key[3:])] = set(tuple(v) for v in value)
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, value in data.items():
            grp.create_dataset('tri{:03d}'.format(key), data=np.array([list(v) for v in value]))


class HDF5TimepointPerimeterCache(BaseHDF5Cache):
    """ Timpoint Perimeter Cache """

    def _from_hdf5(self, db):
        data = {}
        for key, subgroup in db[self.hdf_attr].items():
            perimeters = []
            for value in subgroup.values():
                perimeters.append(list(value))
            data[int(key[1:])] = perimeters
        return data

    def _to_hdf5(self, db, data):
        grp = db.create_group(self.hdf_attr)
        for key, value in data.items():
            subgrp = grp.create_group('p{:03d}'.format(key))
            for i, v in enumerate(value):
                subgrp.create_dataset('{:d}'.format(i), data=np.array(v))

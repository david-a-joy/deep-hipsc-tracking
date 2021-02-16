
# 3rd party
import numpy as np

import h5py

# Our own imports
from ..helpers import FileSystemTestCase

from deep_hipsc_tracking.tracking import persistence_data

# Constants

PERSISTENCE_ATTRS = [
    'tt', 'xx', 'yy', 'time_scale',
    'sm_tt', 'sm_xx', 'sm_yy',
    'sm_dt', 'sm_dx', 'sm_dy',
    'sm_ds', 'sm_dv', 'sm_dtheta',
    'sm_unwrap_dtheta',
    'pct_persistent', 'pct_quiescent',
    'times', 'gap_times', 'speeds', 'distances', 'displacements',
    'timeline', 'waveform', 'mask',
]


# Tests


class TestPersistenceData(FileSystemTestCase):

    def assertPersistenceAttrsEqual(self, obj1, obj2):

        for attr in PERSISTENCE_ATTRS:
            self.assertTrue(hasattr(obj1, attr), msg='obj1 missing {}'.format(attr))
            self.assertTrue(hasattr(obj2, attr), msg='obj2 missing {}'.format(attr))

            val1 = getattr(obj1, attr)
            val2 = getattr(obj2, attr)

            msg = '"{}" mismatch: obj1.{}={} obj2.{}={}'.format(attr, attr, val1, attr, val2)

            try:
                if hasattr(val1, 'dtype') and hasattr(val2, 'dtype'):
                    np.testing.assert_almost_equal(val1, val2, err_msg=msg)
                else:
                    self.assertEqual(val1, val2, msg=msg)
            except Exception:
                print(msg)
                raise

    def test_refuses_to_process_too_short_of_track(self):

        tt = np.linspace(1, 10, 5)
        xx = np.linspace(-10, 5, tt.shape[0])
        yy = np.linspace(-5, 15, tt.shape[0])
        time_scale = 3.0

        obj = persistence_data.calc_track_persistence(
            tt=tt, xx=xx, yy=yy, time_scale=time_scale,
            resample_factor=2,
            smooth_points=0,
            interp_points=3,
            min_persistence_points=6)

        self.assertIsNone(obj)

        obj = persistence_data.calc_track_persistence(
            tt=tt, xx=xx, yy=yy, time_scale=time_scale,
            resample_factor=2,
            smooth_points=0,
            interp_points=3,
            min_persistence_points=5)

        self.assertIsNotNone(obj)

    def test_has_helpful_summary_attributes(self):

        tt = np.linspace(1, 10, 5)
        xx = np.linspace(-10, 5, tt.shape[0])
        yy = np.linspace(-5, 15, tt.shape[0])
        yy[3:] *= 5.0

        time_scale = 3.0
        space_scale = 2.0

        obj = persistence_data.calc_track_persistence(
            tt=tt, xx=xx, yy=yy,
            time_scale=time_scale,
            space_scale=space_scale,
            resample_factor=2,
            smooth_points=0,
            interp_points=3,
            min_persistence_points=1)

        np.testing.assert_almost_equal(obj.duration, 27.0)
        np.testing.assert_almost_equal(obj.distance, 172.0807, decimal=3)
        np.testing.assert_almost_equal(obj.distance, np.sum(obj.sm_ds)*space_scale, decimal=3)
        np.testing.assert_almost_equal(obj.displacement, 169.3451, decimal=3)
        np.testing.assert_almost_equal(obj.disp_to_dist, 0.9841, decimal=3)
        np.testing.assert_almost_equal(obj.average_velocity, 6.2720, decimal=3)
        np.testing.assert_almost_equal(obj.average_speed, 6.3734, decimal=3)

    def test_smooth_values_interp_properly(self):

        tt = np.linspace(1, 10, 10)
        xx = np.linspace(-10, 10, tt.shape[0])
        yy = np.linspace(-5, 15, tt.shape[0])
        time_scale = 3.0

        obj1 = persistence_data.PersistenceData(
            tt=tt, xx=xx, yy=yy, time_scale=time_scale
        )
        obj1.resample_positions(resample_factor=2,
                                smooth_points=0)

        exp_tt = np.linspace(1, 10, 20)
        exp_xx = np.linspace(-10, 10, 20)
        exp_yy = np.linspace(-5, 15, 20)

        np.testing.assert_almost_equal(exp_tt, obj1.sm_tt)
        np.testing.assert_almost_equal(exp_xx, obj1.sm_xx)
        np.testing.assert_almost_equal(exp_yy, obj1.sm_yy)

    def test_smooth_values_with_rolling_mean(self):

        tt = np.linspace(1, 10, 5)
        xx = np.linspace(-10, 5, tt.shape[0])
        yy = np.linspace(-5, 15, tt.shape[0])
        time_scale = 3.0

        obj1 = persistence_data.PersistenceData(
            tt=tt, xx=xx, yy=yy, time_scale=time_scale
        )
        obj1.resample_positions(resample_factor=2,
                                smooth_points=4)

        exp_tt = np.linspace(1, 10, 10)
        exp_xx = np.array([-8.75, -7.9166667, -6.875, -5.625, -4.1666667,
                           -2.5, -0.8333333, 0.625, 1.875, 2.9166667])
        exp_yy = np.array([-3.3333333, -2.2222222, -0.8333333, 0.8333333, 2.7777778,
                           5.0, 7.2222222, 9.1666667, 10.8333333, 12.2222222])

        np.testing.assert_almost_equal(exp_tt, obj1.sm_tt)
        np.testing.assert_almost_equal(exp_xx, obj1.sm_xx)
        np.testing.assert_almost_equal(exp_yy, obj1.sm_yy)

    def test_hdf5_file_roundtrips(self):

        hdf5_file = self.tempdir / 'test.h5'

        tt = np.arange(1, 101)
        xx = np.linspace(-10, 10, tt.shape[0])
        yy = np.linspace(-5, 15, tt.shape[0])
        time_scale = 3.0

        obj1 = persistence_data.PersistenceData(
            tt=tt, xx=xx, yy=yy, time_scale=time_scale
        )
        obj1.resample_positions(resample_factor=2)
        obj1.find_regions(interp_points=2, cutoff=2.5)

        assert not hdf5_file.is_file()

        obj1.to_hdf5(hdf5_file)

        assert hdf5_file.is_file()

        obj2 = persistence_data.PersistenceData.from_hdf5(hdf5_file)

        self.assertPersistenceAttrsEqual(obj1, obj2)

    def test_save_to_subgroup(self):

        hdf5_file = self.tempdir / 'test.h5'
        db = h5py.File(str(hdf5_file), 'w')

        grp = db.create_group('p001')

        tt = np.arange(1, 101)
        xx = np.linspace(-10, 10, tt.shape[0])
        yy = np.linspace(-5, 15, tt.shape[0])
        time_scale = 3.0

        obj1 = persistence_data.PersistenceData(
            tt=tt, xx=xx, yy=yy, time_scale=time_scale
        )
        obj1.resample_positions(resample_factor=2)
        obj1.find_regions(interp_points=2, cutoff=2.5)

        obj1.to_hdf5(grp)
        db.close()

        assert hdf5_file.is_file()

        db = h5py.File(hdf5_file, 'r')
        grp = db['p001']

        obj2 = persistence_data.PersistenceData.from_hdf5(grp)

        db.close()

        self.assertPersistenceAttrsEqual(obj1, obj2)

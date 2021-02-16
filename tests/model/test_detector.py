# Imports

# Standard lib
import json
import pathlib
import unittest

# 3rd Party
import numpy as np

# Our own imports
from .. import helpers

from deep_hipsc_tracking.utils import load_train_test_split
from deep_hipsc_tracking.model._keras import _import_keras
from deep_hipsc_tracking.model import detector, preproc

Input = _import_keras('layers.Input')
Convolution2D = _import_keras('layers.convolutional.Convolution2D')
Model = _import_keras('models.Model')
Adam = _import_keras('optimizers.Adam')


# Helpers


class FakeSampler(preproc.BaseSampler):

    x_shape = (64, 32, 3)
    y_shape = (10, )

    @classmethod
    def data_finder(self, datadir):
        return [pathlib.Path(str(i)) for i in range(1000)]

    @classmethod
    def mask_finder(self, datadir):
        return {str(i): pathlib.Path(str(i)) for i in range(1000)}

    def resample_all(self, num_images):
        x = np.random.ranf((num_images, ) + self.x_shape)
        y = np.random.ranf((num_images, ) + self.y_shape)
        return x, y


class FakeModel(detector.DetectorBase):

    def __init__(self, **kwargs):
        self.roi_outfile = pathlib.Path('fake.hdf5')
        self.is_new = True

        for key, val in kwargs.items():
            setattr(self, key, val)

    def make_detector_fake(self):

        input_net = Input(shape=(32, 32, 3))

        net = Convolution2D(4, (16, 16))(input_net)

        detector = Model(input_net, net)
        detector.compile(loss='mae', optimizer=Adam(lr=1e-4))
        self.detector = detector

    def make_sampler(self, datadir, x_shape, y_shape):

        sampler = preproc.ImageResampler()
        sampler.set_data_loader(datadir=datadir,
                                data_finder=FakeSampler.data_finder)
        sampler.set_mask_loader(mask_finder=FakeSampler.mask_finder)
        sampler.load_files()
        sampler.calc_train_test_split()
        sampler.load_samplers(sampler_type=FakeSampler)

        self.x_train = sampler.train_data
        self.x_test = sampler.test_data
        self.x_validation = sampler.validation_data

        self.x_train.x_shape = x_shape
        self.x_train.y_shape = y_shape
        self.x_test.x_shape = x_shape
        self.x_test.y_shape = y_shape
        self.x_validation.x_shape = x_shape
        self.x_validation.y_shape = y_shape

        self.x_train_shape = x_shape


# Tests


class TestDetectorBase(helpers.FileSystemTestCase):

    def test_get_detectors(self):

        res = FakeModel.get_detectors()
        exp = {'fake': FakeModel.make_detector_fake}

        self.assertEqual(res, exp)

    def test_sample_data(self):

        net = FakeModel()
        net.make_sampler(datadir=self.tempdir,
                         x_shape=(43, 29, 1),
                         y_shape=(7, ))

        x, y = net.sample_data(size=157)

        self.assertEqual(x.shape, (157, 43, 29, 1))
        self.assertEqual(y.shape, (157, 7))

    @unittest.skipIf(Input is None, 'keras not installed')
    def test_can_save_model(self):

        snapshot_config = self.tempdir / 'config.json'

        net = FakeModel()
        net.make_detector()
        net.save_model(snapshot_config)

        self.assertTrue(snapshot_config.is_file())

        with snapshot_config.open('rt') as fp:
            snapshot_data = json.load(fp)

        # Snapshot data can change from version to version
        exp_layers = [
            {
                'class_name': 'InputLayer',
                'config': {
                    'batch_input_shape': [None, 32, 32, 3],
                }
            },
            {
                'class_name': 'Conv2D',
                'config': {
                    'filters': 4,
                    'kernel_size': [16, 16],
                }
            },
        ]

        self.assertEqual(len(snapshot_data['layers']),
                         len(exp_layers))
        for layer, exp_layer in zip(snapshot_data['layers'], exp_layers):
            for key in exp_layer:
                if key == 'config':
                    continue
                self.assertEqual(layer[key], exp_layer[key])
            # Only test stable keys in the config
            for conf_key, exp_val in exp_layer['config'].items():
                self.assertEqual(layer['config'][conf_key],
                                 exp_val)

    @unittest.skipIf(Input is None, 'keras not installed')
    def test_can_save_load_model(self):

        snapshot_dir = self.tempdir / 'snapshot'
        snapshot_config = snapshot_dir / 'config.json'
        snapshot_loss = snapshot_dir / 'losses.json'
        snapshot_weights = snapshot_dir / 'single_cell_weights.hdf5'

        net1 = FakeModel()
        net1.make_detector()
        net1.save_snapshot(snapshot_dir)

        self.assertTrue(snapshot_config.is_file())
        self.assertTrue(snapshot_loss.is_file())
        self.assertTrue(snapshot_weights.is_file())

        self.assertFalse(net1.roi_outfile.is_file())

        net2 = FakeModel.from_snapshot(snapshot_config,
                                       learning_rate=1e-5)

        # Make sure we get the same layers back in the same order
        layers1 = net1.detector.layers
        layers2 = net2.detector.layers

        # TODO: Make this test better?
        self.assertEqual(len(layers1), len(layers2))
        for l1, l2 in zip(layers1, layers2):
            self.assertEqual(type(l1), type(l2))

    @unittest.skipIf(Input is None, 'keras not installed')
    def test_can_save_load_model_train_test_split(self):

        rootdir = self.tempdir / 'FakeData'
        rootdir.mkdir(exist_ok=True, parents=True)
        for i in range(10):
            (rootdir / '{:03d}dots.png'.format(i)).touch()
            (rootdir / '{:03d}cell.png'.format(i)).touch()

        def data_finder(datadir):
            return [p for p in datadir.iterdir() if p.name.endswith('cell.png')]

        def mask_finder(maskdir):
            return {p.stem.replace('dots', 'cell'): p for p in maskdir.iterdir()
                    if p.name.endswith('dots.png')}

        snapshot_dir = self.tempdir / 'a' / 'snapshot'
        snapshot_config = snapshot_dir / 'config.json'
        train_test_split_file1 = snapshot_dir / 'train_test_split.json'
        losses_file = snapshot_dir / 'losses.json'
        weight_file = snapshot_dir / 'single_cell_weights.hdf5'

        model_params = {
            'rootdir': rootdir,
            'data_finder': data_finder,
            'mask_finder': mask_finder,
            'mask_type': 'file',
            'img_rows': 256,
            'img_cols': 256,
            'img_colors': 1,
            'sampler_type': 'complete',
            'sample_with_replacement': True,
            'batch_size': 1,
            'data_cache_size': 0,
            'size_range': (32, 32),
            'rotation_range': (0, 0),
            'flip_horizontal': False,
            'flip_vertical': False,
            'zero_padding': (0, 0),
            'learning_rate': 1e-4,
            'roi_outfile': (self.tempdir / 'single_cell_weights.hdf5'),
        }

        net1 = FakeModel(**model_params)
        net1.load()
        self.assertFalse(snapshot_dir.is_dir())

        net1.save_snapshot(snapshot_dir)

        self.assertTrue(snapshot_config.is_file())
        self.assertTrue(train_test_split_file1.is_file())
        self.assertTrue(losses_file.is_file())
        self.assertTrue(weight_file.is_file())

        train_test_split_file2 = self.tempdir / 'b' / 'snapshot' / 'train_test_split.json'

        net2 = FakeModel.from_snapshot(snapshot_dir, **model_params)
        net2.save_train_test_split(train_test_split_file2)

        self.assertTrue(train_test_split_file2.is_file())

        split1 = load_train_test_split(self.tempdir / 'a')
        split2 = load_train_test_split(self.tempdir / 'b')

        # Make sure we get the same split (in no particular order)
        self.assertEqual(split1.keys(), split2.keys())
        for key in split1:
            set1 = set(split1[key])
            set2 = set(split2[key])
            self.assertEqual(set1, set2)

    @unittest.skipIf(Input is None, 'keras not installed')
    def test_default_detector(self):

        net = detector.DetectorBase()
        with self.assertRaises(AttributeError):
            net.make_detector()

        net = FakeModel()
        net.make_detector()

        self.assertTrue(hasattr(net, 'detector'))

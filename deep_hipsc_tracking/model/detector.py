""" Tools for building detector style neural nets

Subclass :py:class:`DetectorBase` implement :py:func:`make_detector`

API Documentation
-----------------

"""

# Imports

# Standard lib
import time
import json
import pathlib
import datetime
import traceback
import inspect
from typing import Optional, Dict, Callable

# 3rd party
import numpy as np

# Our own imports
from ._keras import _import_keras
from .preproc import ImageResampler

# Mock out keras imports if needed
Adam = _import_keras('optimizers.Adam')
Model = _import_keras('models.Model')

# Constants
BATCH_SIZE = 128

# Helpers


def fix_json(obj):
    """ Mostly convert numpy numbers to python numbers """
    if hasattr(obj, 'dtype'):
        for try_type in (int, float):
            try:
                if np.issubdtype(obj.dtype, try_type):
                    return try_type(obj)
            except TypeError:
                traceback.print_exc()
                continue
        print(obj, type(obj))
        raise ValueError(f'Cannot encode numpy obj: {obj}')
    else:
        print(obj, type(obj))
        raise ValueError(f'Cannot encode {obj}')


def find_snapshot(snapshot_dir: pathlib.Path,
                  snapshot_prefix: str = '',
                  snapshot_root: Optional[pathlib.Path] = None):
    """ Find the snapshot directory

    :param Path snapshot_dir:
        The path to the snapshot directory or None to go fishing
    :param str snapshot_prefix:
        The model-specific snapshot prefix
    :param Path snapshot_root:
        The base directory where snapshots are stored
    :returns:
        The path to the snapshot directory or None if one can't be found
    """

    if snapshot_dir is None:
        return None

    snapshot_dir = pathlib.Path(snapshot_dir)
    if snapshot_dir.is_dir():
        return snapshot_dir

    if snapshot_root is None:
        return None

    snapshot_name = snapshot_dir.name.strip('-')

    # Try adding a prefix to things
    for prefix in ('', snapshot_prefix, 'snapshot-' + snapshot_prefix):
        if prefix != '':
            prefix += '-'
        try_dir = snapshot_root / (prefix + snapshot_name)
        if try_dir.is_dir():
            return try_dir
    return None


# Classes


class DetectorBase(object):
    """ Base class for detector objects

    Subclass this and implement `make_detector`, which creates a keras Model
    and assigns it to the `detector` attribute of the class.
    """

    model_name = None

    @classmethod
    def from_snapshot(cls, snapshot_dir: pathlib.Path, **kwargs):
        """ Load the class from a snapshot

        :param Path snapshot_dir:
            The snapshot directory to load
        :param \\*\\*kwargs:
            The keyword args to pass to the constructor
        """

        net = cls(**kwargs)
        net.is_new = False

        snapshot_dir = pathlib.Path(snapshot_dir)
        snapshot_dir = snapshot_dir.resolve()
        if snapshot_dir.is_file():
            snapshot_dir = snapshot_dir.parent

        # Have to load the optimizer before the config
        net.load_optimizer()
        net.load_detector_from_snapshot(snapshot_dir / 'config.json',
                                        detector=kwargs.get('detector'))

        loss_function = getattr(net, 'loss_function', 'mae')
        net.detector.compile(loss=loss_function,
                             optimizer=net.opt)
        net.detector.summary()

        net.weight_outfile = snapshot_dir / 'single_cell_weights.hdf5'
        print(f'Loading snapshot weights: {net.weight_outfile}')
        net.load_weights(net.weight_outfile)

        if hasattr(net, 'rootdir') and hasattr(net, 'data_finder'):
            net.load_data(snapshot_dir=snapshot_dir)
        return net

    def load(self, snapshot_dir: Optional[pathlib.Path] = None):
        """ Initialize the detectory model

        :param Path snapshot_dir:
            If not None, reload the train/test split from the snapshot
        """
        self.load_data(snapshot_dir=snapshot_dir)
        self.load_optimizer()
        self.make_detector()
        if getattr(self, 'weight_outfile', None) is None:
            if snapshot_dir is not None:
                self.weight_outfile = snapshot_dir / 'single_cell_weights.hdf5'
            else:
                self.weight_outfile = None
        self.load_weights(weight_outfile=self.weight_outfile)

    def load_detector_from_snapshot(self,
                                    snapshot_config: pathlib.Path,
                                    detector: Optional[str] = None):
        """ Reload the Keras config file or build the model from scratch

        :param Path snapshot_config:
            The JSON encoded config file to load
        :param str detector:
            The detector name to load
        """
        print(f'Loading config for detector: {detector}')
        # Try to load the net from the snapshot
        if snapshot_config.is_file():
            print(f'Loading snapshot config: {snapshot_config}')

            with snapshot_config.open('rt') as fp:
                config = json.load(fp)

            try:
                self.detector = Model.from_config(config)
            except ValueError:
                print(f'Snapshot config seems corrupted: {snapshot_config}')
                print(f'Trying to regenerate "{detector}" from scratch...')
                self.make_detector(detector=detector)
        else:
            print(f'No snapshot config found at: {snapshot_config}')
            print(f'Trying to regenerate "{detector}" from scratch...')
            self.make_detector(detector=detector)
        assert self.detector is not None

    def load_data(self, snapshot_dir: Optional[pathlib.Path] = None):
        """ Load the actual data

        :param Path snapshot_dir:
            If not None, reload the train/test split from the snapshot
        """

        if snapshot_dir is None:
            split_file = None
        else:
            split_file = pathlib.Path(snapshot_dir) / 'train_test_split.json'

        sampler = ImageResampler()
        sampler.set_data_loader(datadir=self.rootdir,
                                data_finder=self.data_finder)
        sampler.set_mask_loader(masks=None,
                                mask_finder=self.mask_finder,
                                mask_type=self.mask_type)
        sampler.load_files()

        # Cache the train/test split for reproducibility
        if split_file is None:
            print('Generating train/test split...')
            sampler.calc_train_test_split()
        elif split_file.is_file():
            print(f'Reloading train/test split from {split_file}')
            sampler.load_train_test_split(split_file)
        else:
            print('Generating train/test split...')
            sampler.calc_train_test_split()
            print(f'Saving train/test split to {split_file}')
            sampler.save_train_test_split(split_file)

        # Make sure we got a valid split for the loader
        if not sampler.is_split_under_datadir(self.rootdir):
            print('Old split is for a different dataset, forcing new split...')
            sampler.calc_train_test_split()

        sampler.load_samplers(sampler_type=self.sampler_type,
                              sample_with_replacement=self.sample_with_replacement,
                              image_layout='tensorflow',
                              input_shape=(self.img_rows, self.img_cols, self.img_colors),
                              batch_size=self.batch_size,
                              cache_size=self.data_cache_size,
                              size_range=self.size_range,
                              rotation_range=self.rotation_range,
                              flip_horizontal=self.flip_horizontal,
                              flip_vertical=self.flip_vertical,
                              zero_padding=self.zero_padding)

        self.sampler = sampler
        self.x_train = sampler.train_data
        self.x_test = sampler.test_data
        self.x_validation = sampler.validation_data
        self.x_train_shape = (self.img_rows, self.img_cols, self.img_colors)

    def load_optimizer(self, learning_rate: Optional[float] = None):
        """ Load the optimizer for this class

        :param float learning_rate:
            If not None, the new learning rate (else use the class rate)
        """
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.opt = Adam(learning_rate=learning_rate)

    def sample_data(self, size: int = BATCH_SIZE, mode: str = 'train'):
        """ Create a vector of samples

        :param int size:
            The size of the batch
        :param str mode:
            Which bucket (train/test/etc) to pull samples from
        """

        # Sample from the training data
        if mode == 'train':
            self.x_train.batch_size = size
            x_real, y_real = next(self.x_train)
        elif mode == 'test':
            self.x_test.batch_size = size
            x_real, y_real = next(self.x_test)
        else:
            raise KeyError(f'Unknown mode {mode}')

        y_final = self.reshape_y_data(y_real)

        if getattr(self, 'sampler_type', None) == 'conv':
            assert x_real.shape[0] == size
        else:
            assert x_real.shape == (size, ) + self.x_train_shape
        assert y_final.shape[0] == size
        return x_real, y_final

    def set_opt_lr(self, opt: float):
        """ Set the loss rates for the optimizer

        :param float opt:
            The optimization rate to use
        """
        self.opt.lr = opt

    def have_weights(self, weight_outfile: Optional[pathlib.Path] = None):
        """ Return whether or not we have pre-trained weights

        :param Path weight_outfile:
            The path to search for weights under
        """
        if weight_outfile is None:
            weight_outfile = self.weight_outfile
        return weight_outfile.is_file()

    def load_weights(self, weight_outfile: Optional[pathlib.Path] = None):
        """ Load the pre-trained weights for the net

        :param Path weight_outfile:
            If not None, the weight file to load from
        """
        if self.is_new:
            print('New net, ignoring load weights...')
            return

        if weight_outfile is None:
            if self.weight_outfile.is_file():
                weight_outfile = self.weight_outfile
        if weight_outfile.is_file():
            print(f'Loading old weights from {weight_outfile}')
            self.detector.load_weights(str(weight_outfile))
        else:
            raise OSError(f'Cannot find weight file: {weight_outfile}')

    def save_weights(self, weight_outfile: Optional[pathlib.Path] = None):
        """ Save the weights for the net

        :param Path weight_outfile:
            If not None, the weight file to save to
        """
        if weight_outfile is None:
            weight_outfile = self.weight_outfile
        print(f'Saving weights to {weight_outfile}')
        weight_outfile.parent.mkdir(exist_ok=True, parents=True)
        self.detector.save_weights(str(weight_outfile))

    def save_model(self, snapshot_config: pathlib.Path):
        """ Save the metadata for the model

        :param Path snapshot_config:
            The path to the configuration file to save to
        """
        snapshot_config.parent.mkdir(exist_ok=True, parents=True)
        print(f'Saving model to {snapshot_config}')
        with snapshot_config.open('wt') as fp:
            config = self.detector.get_config()
            json.dump(config, fp, default=fix_json)

    def save_losses(self, loss_file: pathlib.Path):
        """ Save the loss profile for the training

        :param Path loss_file:
            The path to the loss file to save to
        """

        train_losses = getattr(self, 'train_losses', [])
        test_losses = getattr(self, 'test_losses', [])

        train_start_time = getattr(self, 'train_start_time', None)
        train_end_time = time.time()

        # Save the train/test split we used
        train_files = [p.name for p in getattr((self, 'x_train', None), 'files', [])]
        test_files = [p.name for p in getattr((self, 'x_test', None), 'files', [])]

        config = {
            'train_files': train_files,
            'test_files': test_files,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_start_time': train_start_time,
            'train_end_time': train_end_time,
        }

        print(f'Saving losses to {loss_file}')
        loss_file.parent.mkdir(exist_ok=True, parents=True)
        with loss_file.open('wt') as fp:
            json.dump(config, fp)

    def save_train_test_split(self, split_file: pathlib.Path):
        """ Save the train test split to a file

        :param Path split_file:
            The path to the split file to save to
        """
        sampler = getattr(self, 'sampler', None)
        if sampler is None:
            print('No data sampler loaded...')
            return
        print(f'Saving train/test split to {split_file}')
        sampler.save_train_test_split(split_file)

    def save_snapshot(self, snapshot_dir: Optional[pathlib.Path] = None):
        """ Save a snapshot

        :param Path snapshot_dir:
            If not None, the directory to save an entire model snapshot to
        """
        self.is_new = False
        print('Saving snapshot...')

        if snapshot_dir is None:
            snapshot_dir = pathlib.Path(self.snapshot_dir)

            snapshot_time = datetime.datetime.now()
            snapshot_time = snapshot_time.strftime('%Y%m%d-%H%M%S')
            if self.model_name is None:
                snapshot_name = 'snapshot-' + snapshot_time
            else:
                snapshot_name = 'snapshot-' + self.model_name + '-' + snapshot_time
            snapshot_dir = snapshot_dir / snapshot_name
        else:
            snapshot_dir = pathlib.Path(snapshot_dir)

        print(f'Snapshot directory: {snapshot_dir}')

        model_file = snapshot_dir / 'config.json'
        weight_file = snapshot_dir / 'single_cell_weights.hdf5'
        loss_file = snapshot_dir / 'losses.json'
        split_file = snapshot_dir / 'train_test_split.json'

        # Snapshot
        snapshot_dir.mkdir(parents=True)
        self.save_weights(weight_file)
        self.save_model(model_file)
        self.save_losses(loss_file)
        self.save_train_test_split(split_file)

    def train_for_n(self, nb_epoch: int = 1,
                    batch_size: Optional[int] = None):
        """ Train for n epochs

        :param int nb_epoch:
            The number of epochs to train for
        :param int batch_size:
            The number of individual samples to draw each epoch
        """

        if batch_size is None:
            batch_size = self.batch_size

        save_steps = self.save_steps
        eval_steps = self.eval_steps
        snapshot_steps = self.snapshot_steps

        if nb_epoch < 1:
            return

        train_losses = getattr(self, 'train_losses', [])
        self.train_losses = train_losses

        t0 = time.time()
        self.train_start_time = t0

        for i in range(nb_epoch):
            print('{} Epoch {} of {} {}'.format('='*5, i + 1, nb_epoch, '='*5))

            # Train the discriminator for an epoch
            print('Training detector...')
            x, y = self.sample_data(size=batch_size)

            d_loss = self.detector.train_on_batch(x, y)
            print(f'Loss: {d_loss}')

            train_losses.append((i, float(d_loss)))

            if i % save_steps == 0:
                self.save_weights()

            if i > 0 and i % eval_steps == 0:
                t1 = time.time()
                print(f'{t1 - t0:0.2f} secs for {eval_steps} steps')
                t0 = t1
                self.eval_model(epoch=i)

            if i > 0 and i % snapshot_steps == 0:
                self.save_snapshot()

        if i % save_steps != 0:
            self.save_weights()

    def eval_model(self, batch_size: Optional[int] = None,
                   epoch: Optional[int] = None):
        """ Evaluate the model

        :param int batch_size:
            The number of draws to evaluate the model on
        :param int epoch:
            The current epoch number
        """

        if batch_size is None:
            batch_size = self.validation_batch_size

        x, y = self.sample_data(size=batch_size,
                                mode='test')
        score = self.detector.evaluate(x, y, batch_size=batch_size)
        print(f'Test loss: {score}')

        self.test_losses = getattr(self, 'test_losses', [])
        self.test_losses.append((epoch, float(score)))

    def reshape_y_data(self, y_real: np.ndarray) -> np.ndarray:
        """ Reshape the y data before using

        If your input and output data have different shapes, implement this function

        :param ndarray y_real:
            The (num_samples x data_shape) array to reshape
        :returns:
            The reshaped array
        """
        return y_real

    @classmethod
    def get_detectors(cls) -> Dict[str, Callable]:
        """ Get all the detectors defined on this class """

        detectors = {}

        for name, meth in inspect.getmembers(cls):
            if not name.startswith('make_detector_'):
                continue
            detector_name = name[len('make_detector_'):]
            detectors[detector_name] = meth
        return detectors

    def make_detector(self, detector=None, **kwargs):
        """ Make a detector

        :param str detector:
            The name of the detector to use
        :param \\*\\*kwargs:
            Options to pass to the detector
        """

        if detector is None:
            detector = getattr(self, 'detector', None)
        available_detectors = self.get_detectors()
        if len(available_detectors) == 0:
            raise AttributeError('Define a detector!')

        if detector is None and len(available_detectors) == 1:
            detector = list(available_detectors.keys())[0]
        return available_detectors[detector](self, **kwargs)

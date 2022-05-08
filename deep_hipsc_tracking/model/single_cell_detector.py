""" Single Cell Detection Deep Neural Nets

The main class :py:class:`SingleCellDetector` implements several different
neural nets to detect single cells:

* :py:meth:`SingleCellDetector.make_detector_unet`: U-Net from Ronneberger et al 2015
* :py:meth:`SingleCellDetector.make_detector_countception`: Count-ception from Cohen et al 2017
* :py:meth:`SingleCellDetector.make_detector_fcrn_a`: fully convolutional regression network A from Xie et al 2015
* :py:meth:`SingleCellDetector.make_detector_fcrn_b`: fully convolutional regression network B from Xie et al 2015
* :py:meth:`SingleCellDetector.make_detector_residual_unet`: Residual U-Net from Xie et al 2018

API Documentation
-----------------

"""

# Standard lib
import json
import time
import shutil
import pathlib
from typing import Optional

# 3rd party
import numpy as np

from scipy.ndimage import distance_transform_edt

import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from skimage.measure import block_reduce

# Our own imports
from ._keras import _import_keras

Input, Activation, Dropout, Lambda = _import_keras(
    'Input', 'Activation', 'Dropout', 'Lambda', module='layers')
BatchNormalization = _import_keras('layers.BatchNormalization')
Convolution2D, ZeroPadding2D, UpSampling2D, Cropping2D = _import_keras(
    'Convolution2D', 'ZeroPadding2D', 'UpSampling2D', 'Cropping2D',
    module='layers')
LeakyReLU, ELU = _import_keras(
    'LeakyReLU', 'ELU', module='layers')
Concatenate, Add = _import_keras(
    'Concatenate', 'Add', module='layers')
MaxPooling2D, AveragePooling2D = _import_keras(
    'MaxPooling2D', 'AveragePooling2D', module='layers')
K = _import_keras('backend')
Model = _import_keras('models.Model')

# ActivationMaximization = _import_keras_vis('losses.ActivationMaximization')
# Optimizer = _import_keras_vis('optimizer.Optimizer')

from ..utils import save_point_csvfile, save_image
from ..plotting import set_plot_style, add_colorbar

from . import DetectorBase, DataFinders, clamp, predict_with_steps

# Constants

THISDIR = pathlib.Path(__file__).resolve().parent

PLOT_STYLE = 'dark'

DETECTOR = 'countception'

COMPOSITE_MODE = 'peak'
COMPOSITE_STRIDE = 32
COMPOSITE_BATCH_SIZE = 12
COMPOSITE_TRANSFORMS = 'none'

IMG_ROWS, IMG_COLS, IMG_COLORS = 256, 256, 1

RELU_LEAKYNESS = 0.1
ELU_ALPHA = 1.0  # FIXME: Is this a good value?

PEAK_SHARPNESS = 8
PEAK_DISTANCE = 3

DROPOUT_RATE = 0.25

RESPONSE_MIN = 0.0
RESPONSE_MAX = 0.8

BATCH_SIZE = 8  # Number of samples in a main batch

SAVE_STEPS = 100  # Number of steps between saves
EVAL_STEPS = 100  # Number of steps between test evaluations
SNAPSHOT_STEPS = 1000  # Number of steps between snapshots

NB_EPOCH = 1000  # Number of steps to train

DATA_CACHE_SIZE = None  # Size of the lru_cache

CUR_OPT_STAGE = 0  # Current optimizer stage

SIZE_RANGE = (32, 32)  # Pixels per side

DATA_RESIZE = 4  # How much to reduce the sampling images by
DETECTION_THRESHOLD = 0.3  # 0.5 for a well trained net
DETECTION_EROSION = 0  # How many pixels of the heatmap to erode

# Optimizer stages
OPT_STAGES = [
    1e-4,
    5e-5,
    2e-5,
    1e-5,
    1e-6,
]

# Activation function for the nets: One of, 'relu', 'leaky_relu', 'elu'
ACTIVATION_FUNCTION = {
    'countception': 'leaky_relu',
    'unet': 'leaky_relu',
    'fcrn_a': 'leaky_relu',
    'fcrn_a_wide': 'leaky_relu',
    'fcrn_b': 'leaky_relu',
    'fcrn_b_wide': 'leaky_relu',
    'residual_unet': 'elu',
    'residual_unet_wide': 'elu',
}
# Input shapes for the nets: (rows, cols, colors)
INPUT_SHAPE = {
    'countception': (256, 256, 1),
    'unet': (256, 256, 1),
    'fcrn_a': (100, 100, 1),
    'fcrn_a_wide': (256, 256, 1),
    'fcrn_b': (100, 100, 1),
    'fcrn_b_wide': (256, 256, 1),
    'residual_unet': (100, 100, 1),
    'residual_unet_wide': (256, 256, 1),
}
# Output shapes for the nets: (rows, cols)
OUTPUT_SHAPE = {
    'countception': (225, 225),
    'unet': (68, 68),
    'fcrn_a': (100, 100),
    'fcrn_a_wide': (256, 256),
    'fcrn_b': (100, 100),
    'fcrn_b_wide': (256, 256),
    'residual_unet': (100, 100),
    'residual_unet_wide': (256, 256),
}


# Classes


class SingleCellDetector(DetectorBase):
    """ Single cell detection object

    :param str data_finder_mode:
        A data set key from DATA_FINDER_MODES
    :param int peak_sharpness:
        How strongly to spread the peak data out from the mask
    :param int peak_distance:
        How far to allow the peak to link
    :param float detection_threshold:
        How strong a detection is needed to call a "cell"
    :param int detection_erosion:
        How many pixels to erode before using the mask
    """

    model_name = 'single-cell'

    def __init__(self, data_finder_mode=DataFinders.default_mode,
                 peak_sharpness=PEAK_SHARPNESS,
                 peak_distance=PEAK_DISTANCE,
                 detection_threshold=DETECTION_THRESHOLD,
                 detection_erosion=DETECTION_EROSION,
                 batch_size=BATCH_SIZE,
                 detector=DETECTOR):
        self.is_new = True

        self.dropout_rate = DROPOUT_RATE

        # Load the detector
        self.detector = self.detector_name = detector

        # Activation function for the detector
        self.activation_function = ACTIVATION_FUNCTION[detector]
        self.activation_cls = {
            'relu': lambda: Activation('relu'),
            'leaky_relu': lambda: LeakyReLU(RELU_LEAKYNESS),
            'elu': lambda: ELU(ELU_ALPHA),
        }[self.activation_function]

        # Load the input shape for each detector
        img_rows, img_cols, img_colors = INPUT_SHAPE[detector]
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_colors = img_colors

        # Load the output shape for each detector
        total_rows, total_cols = OUTPUT_SHAPE[detector]
        self.total_rows = total_rows
        self.total_cols = total_cols

        self.learning_rate = 1e-4

        # Load the selected data finder set
        finders = DataFinders(data_finder_mode)
        self.data_finder_mode = data_finder_mode
        self.data_finder = finders.data_finder
        self.mask_finder = finders.mask_finder
        self.mask_type = finders.mask_type
        self.rootdir = finders.rootdir

        if data_finder_mode in ('training_peaks', 'training_confocal', 'real'):
            self.reshape_y_data = self.reshape_y_data_peaks
        else:
            self.reshape_y_data = self.reshape_y_data_points

        self.data_cache_size = DATA_CACHE_SIZE

        self.batch_size = batch_size
        self.validation_batch_size = batch_size

        self.save_steps = SAVE_STEPS
        self.eval_steps = EVAL_STEPS
        self.snapshot_steps = SNAPSHOT_STEPS

        self.size_range = SIZE_RANGE
        self.rotation_range = (-45, 45)  # degrees
        self.flip_horizontal = True  # randomly flip horizontally
        self.flip_vertical = True  # randomly flip vertically
        self.zero_padding = 32  # How many pixels of zero to add to each side
        self.peak_sharpness = peak_sharpness  # How sharp of a peak to make
        self.peak_distance = peak_distance  # How many pixels between peaks
        self.detection_threshold = detection_threshold  # Level to cut off the detector at
        self.detection_erosion = detection_erosion  # Number of pixels to erode from the detector

        self.sampler_type = 'complete'

        self.sample_with_replacement = True

    def make_detector_unet(self, loss_function='binary_crossentropy'):
        """ Detector based on the on the U-net paper

        ``Ronneberger, O., Fischer, P., and Brox, T. (2015). U-net: Convolutional
        networks for biomedical image segmentation. In International Conference on
        Medical Image Computing and Computer-Assisted Intervention, (Springer),
        pp. 234–241.``
        """

        exp_total_rows, exp_total_cols = self.total_rows, self.total_cols

        input_net = Input(shape=(self.img_rows, self.img_cols, self.img_colors))
        print(f'Initial size: {self.img_rows}x{self.img_cols}')

        def step_down(H, base_size, conv_size):
            H = MaxPooling2D(pool_size=(2, 2),
                             data_format='channels_last')(H)
            H = Convolution2D(conv_size, (3, 3),
                              padding='valid',
                              data_format='channels_last')(H)
            H = self.activation_cls()(H)
            H = Convolution2D(conv_size, (3, 3),
                              padding='valid',
                              data_format='channels_last')(H)
            H = H1 = self.activation_cls()(H)

            base_rows, base_cols = base_size
            new_rows = base_rows // 2 - 4
            new_cols = base_cols // 2 - 4
            return H, H1, (new_rows, new_cols)

        def step_up(H_up, H_across, base_up_size, base_across_size, conv_size, do_upsample=True):

            base_across_rows, base_across_cols = base_across_size
            base_up_rows, base_up_cols = base_up_size

            across_shift_rows = base_across_rows % 2
            across_shift_cols = base_across_cols % 2
            across_crop_rows = (base_across_rows - base_up_rows)//2
            across_crop_cols = (base_across_cols - base_up_cols)//2

            print('Crop: {} + {}'.format((across_crop_rows, across_crop_cols),
                                         (across_shift_rows, across_shift_cols)))
            print('Base size: {} vs {}'.format(base_up_size, base_across_size))

            H_across = Cropping2D(cropping=((across_crop_rows, across_crop_rows+across_shift_rows),
                                            (across_crop_cols, across_crop_cols+across_shift_cols)),
                                  data_format='channels_last')(H_across)
            net = Concatenate(axis=3)([H_up, H_across])
            net = Convolution2D(conv_size, (3, 3),
                                padding='valid',
                                data_format='channels_last')(net)
            net = self.activation_cls()(net)
            net = Convolution2D(conv_size, (3, 3),
                                padding='valid',
                                data_format='channels_last')(net)
            net = self.activation_cls()(net)

            if do_upsample:
                net = UpSampling2D(size=(2, 2),
                                   data_format='channels_last')(net)
                base_size = (base_up_rows - 4) * 2, (base_up_cols - 4) * 2
            else:
                base_size = base_up_rows - 4, base_up_cols - 4
            return net, base_size

        # 1st tier down
        base_rows, base_cols = self.img_rows, self.img_cols
        net = Convolution2D(64, (3, 3),
                            padding='valid',
                            data_format='channels_last')(input_net)
        net = self.activation_cls()(net)

        net = Convolution2D(64, (3, 3),
                            padding='valid',
                            data_format='channels_last')(net)
        net = net1 = self.activation_cls()(net)
        base_size1 = base_rows - 4, base_cols - 4

        # 2nd tier down
        net, net2, base_size2 = step_down(net, base_size1, conv_size=128)

        # 3rd tier down
        net, net3, base_size3 = step_down(net, base_size2, conv_size=256)

        # 4th tier down
        net, net4, base_size4 = step_down(net, base_size3, conv_size=512)

        # Bottom of the net
        net = MaxPooling2D(pool_size=(2, 2),
                           data_format='channels_last')(net)
        net = Convolution2D(1024, (3, 3),
                            padding='valid',
                            data_format='channels_last')(net)
        net = self.activation_cls()(net)

        net = Convolution2D(1024, (3, 3),
                            padding='valid',
                            data_format='channels_last')(net)
        net = self.activation_cls()(net)
        net = UpSampling2D(size=(2, 2),
                           data_format='channels_last')(net)
        base_rows4, base_cols4 = base_size4
        base_rows5 = (base_rows4 // 2 - 4)*2
        base_cols5 = (base_cols4 // 2 - 4)*2
        base_size5 = (base_rows5, base_cols5)

        # 4th tier up
        net, base_size6 = step_up(net, net4, base_size5, base_size4, conv_size=512)

        # 3rd tier up
        net, base_size7 = step_up(net, net3, base_size6, base_size3, conv_size=256)

        # 2nd tier up
        net, base_size8 = step_up(net, net2, base_size7, base_size2, conv_size=128)

        # 1st tier up
        net, base_size9 = step_up(net, net1, base_size8, base_size1, conv_size=64,
                                  do_upsample=False)
        net = Convolution2D(1, (1, 1),
                            padding='same',
                            data_format='channels_last')(net)
        net = self.activation_cls()(net)

        self.total_rows, self.total_cols = base_size9
        print('Got final size: {}x{}'.format(self.total_rows, self.total_cols))
        print('Expected size:  {}x{}'.format(exp_total_rows, exp_total_cols))
        assert (self.total_rows, self.total_cols) == (exp_total_rows, exp_total_cols)

        detector = Model(input_net, net)
        detector.compile(loss=loss_function, optimizer=self.opt)
        detector.summary()

        self.detector = detector

    def make_detector_countception(self, skip_batch_norm=False, loss_function='mae'):
        """ Detector based on the Count-ception architecture

        ``Cohen, J.P., Lo, H.Z., and Bengio, Y. (2017). Count-ception: Counting by
        Fully Convolutional Redundant Counting. arXiv Preprint arXiv:1703.08710.``
        """

        exp_total_rows, exp_total_cols = self.total_rows, self.total_cols

        # Size of the training data
        self.total_rows, self.total_cols = self.img_rows, self.img_cols
        print('Initial size: {}x{}'.format(self.img_rows, self.img_cols))

        def conv_factory(H, num_filter, filter_size, stride=1, pad=0):
            H = ZeroPadding2D(padding=(pad, pad))(H)
            self.total_rows += pad*2
            self.total_cols += pad*2
            H = Convolution2D(num_filter, (filter_size, filter_size),
                              strides=(stride, stride),
                              padding='valid')(H)
            self.total_rows -= filter_size - 1
            self.total_cols -= filter_size - 1
            H = self.activation_cls()(H)
            if not skip_batch_norm:
                H = BatchNormalization()(H)
            return H

        def simple_factory(H, ch_1x1, ch_3x3):
            H_1x1 = conv_factory(H, filter_size=1, pad=0, num_filter=ch_1x1)
            H_3x3 = conv_factory(H, filter_size=3, pad=1, num_filter=ch_3x3)
            return Concatenate(axis=-1)([H_1x1, H_3x3])

        input_net = Input(shape=(self.img_rows, self.img_cols, self.img_colors))

        net = conv_factory(input_net, filter_size=3, num_filter=64, pad=0)
        net = simple_factory(net, 16, 16)
        net = simple_factory(net, 16, 32)
        net = conv_factory(net, filter_size=14, num_filter=16)
        net = simple_factory(net, 112, 48)
        net = simple_factory(net, 64, 32)
        net = simple_factory(net, 40, 40)
        net = simple_factory(net, 32, 96)
        net = conv_factory(net, filter_size=17, num_filter=32)
        net = conv_factory(net, filter_size=1, pad=0, num_filter=64)
        net = conv_factory(net, filter_size=1, pad=0, num_filter=64)
        net = conv_factory(net, filter_size=1, num_filter=1, stride=1)

        print('Got final size: {}x{}'.format(self.total_rows, self.total_cols))
        print('Expected size:  {}x{}'.format(exp_total_rows, exp_total_cols))
        assert (self.total_rows, self.total_cols) == (exp_total_rows, exp_total_cols)

        detector = Model(input_net, net)
        detector.compile(loss=loss_function, optimizer=self.opt)
        detector.summary()

        self.detector = detector

    def make_detector_fcrn_a(self, loss_function='mae'):
        """ Detector based on fully convolutional regression network A

        This network is pure 3x3 convolutions, 2x2 pooling, 2x2 upsampling

        Xie, W., Noble, J.A., and Zisserman, A. (2016). Microscopy cell counting
        and detection with fully convolutional regression networks.
        Computer Methods in Biomechanics and Biomedical Engineering: Imaging &
        Visualization 1–10.
        """

        exp_total_rows, exp_total_cols = self.total_rows, self.total_cols
        self.total_rows, self.total_cols = self.img_rows, self.img_cols

        def downsample(H, num_filter, filter_size, pool_size):
            H = Convolution2D(num_filter, (filter_size, filter_size),
                              padding='same')(H)
            H = self.activation_cls()(H)
            H = MaxPooling2D(pool_size=(pool_size, pool_size))(H)
            pad_topleft = self.total_rows % 2
            self.total_rows //= 2
            self.total_cols //= 2
            return H, pad_topleft

        def upsample(H, num_filter, filter_size, upsample_size, pad_topleft=False):
            # FIXME: Change after https://github.com/keras-team/keras/pull/9303
            H = UpSampling2D(size=(2, 2),
                             #interpolation='bilinear',
                             data_format='channels_last')(H)
            self.total_rows *= 2
            self.total_cols *= 2
            if pad_topleft:
                # Needed to get back to 25x25
                H = ZeroPadding2D(padding=((1, 0), (1, 0)))(H)
                self.total_rows += 1
                self.total_cols += 1
            H = Convolution2D(num_filter, (filter_size, filter_size),
                              padding='same')(H)
            H = self.activation_cls()(H)
            return H

        input_net = Input(shape=(self.img_rows, self.img_cols, self.img_colors))

        # Downsample 50x50
        net, pad_topleft1 = downsample(input_net, num_filter=32, filter_size=3, pool_size=2)

        # Downsample 25x25
        net, pad_topleft2 = downsample(net, num_filter=64, filter_size=3, pool_size=2)

        # Downsample 12x12
        net, pad_topleft3 = downsample(net, num_filter=128, filter_size=3, pool_size=2)

        # Bottom layer (12x12)
        net = Convolution2D(512, (3, 3),
                            padding='same')(net)
        net = self.activation_cls()(net)

        # Upsample 25x25
        net = upsample(net, num_filter=128, filter_size=3, upsample_size=2, pad_topleft=pad_topleft3)

        # Upsample 50x50
        net = upsample(net, num_filter=64, filter_size=3, upsample_size=2, pad_topleft=pad_topleft2)

        # Upsample 100x100
        net = upsample(net, num_filter=32, filter_size=3, upsample_size=2, pad_topleft=pad_topleft1)

        # Final layer
        net = Convolution2D(1, (3, 3),
                            padding='same')(net)
        net = self.activation_cls()(net)

        print('Got final size: {}x{}'.format(self.total_rows, self.total_cols))
        print('Expected size:  {}x{}'.format(exp_total_rows, exp_total_cols))
        assert (self.total_rows, self.total_cols) == (exp_total_rows, exp_total_cols)

        detector = Model(input_net, net)
        detector.compile(loss=loss_function, optimizer=self.opt)
        detector.summary()

        self.detector = detector

    def make_detector_fcrn_b(self, loss_function='mae'):
        """ Detector based on fully convolutional regression network B

        This network uses 3x3 convs and 2x2 pooling to downsample,
        5x5 convs and 2x2 upsampling to upsample.

        Pooling alternates every other layer.

        Xie, W., Noble, J.A., and Zisserman, A. (2016). Microscopy cell counting
        and detection with fully convolutional regression networks.
        Computer Methods in Biomechanics and Biomedical Engineering: Imaging &
        Visualization 1–10.
        """
        exp_total_rows, exp_total_cols = self.total_rows, self.total_cols
        self.total_rows, self.total_cols = self.img_rows, self.img_cols

        def downsample(H, num_filter, filter_size, pool_size):
            H = Convolution2D(num_filter, (filter_size, filter_size),
                              padding='same')(H)
            H = self.activation_cls()(H)
            if pool_size > 1:
                # FIXME: Should this be parameterized by the actual pool size?
                pad_topleft = self.total_rows % 2
                H = MaxPooling2D(pool_size=(pool_size, pool_size))(H)
                self.total_rows //= 2
                self.total_cols //= 2
            else:
                pad_topleft = 0
            return H, pad_topleft

        def upsample(H, num_filter, filter_size, upsample_size, pad_topleft=False):
            # FIXME: Change after https://github.com/keras-team/keras/pull/9303
            H = UpSampling2D(size=(2, 2),
                             #interpolation='bilinear',
                             data_format='channels_last')(H)
            self.total_rows *= 2
            self.total_cols *= 2
            if pad_topleft:
                # Needed to get back to 25x25
                H = ZeroPadding2D(padding=((1, 0), (1, 0)))(H)
                self.total_rows += 1
                self.total_cols += 1
            H = Convolution2D(num_filter, (filter_size, filter_size),
                              padding='same')(H)
            H = self.activation_cls()(H)
            return H

        input_net = Input(shape=(self.img_rows, self.img_cols, self.img_colors))

        # Downsample 50x50
        net, _ = downsample(input_net, num_filter=32, filter_size=3, pool_size=1)
        net, pad_topleft1 = downsample(net, num_filter=64, filter_size=3, pool_size=2)

        # Downsample 25x25
        net, _ = downsample(net, num_filter=128, filter_size=3, pool_size=1)
        net, pad_topleft2 = downsample(net, num_filter=256, filter_size=3, pool_size=2)

        # Bottom layer, 25x25
        net = Convolution2D(256, (5, 5),
                            padding='same')(net)
        net = self.activation_cls()(net)

        # Upsample 50x50
        net = upsample(net, num_filter=256, filter_size=5, upsample_size=2, pad_topleft=pad_topleft2)

        # Upsample to final layer
        net = upsample(net, num_filter=1, filter_size=5, upsample_size=2, pad_topleft=pad_topleft2)

        print('Got final size: {}x{}'.format(self.total_rows, self.total_cols))
        print('Expected size:  {}x{}'.format(exp_total_rows, exp_total_cols))
        assert (self.total_rows, self.total_cols) == (exp_total_rows, exp_total_cols)

        detector = Model(input_net, net)
        detector.compile(loss=loss_function, optimizer=self.opt)
        detector.summary()

        self.detector = detector

    def make_detector_residual_unet(self, loss_function='mae'):
        """ Make a fully residual CNN cell detector

        Xie, Y., Xing, F., Shi, X., Kong, X., Su, H., and Yang, L. (2018).
        Efficient and robust cell detection: A structured regression approach.
        Medical Image Analysis 44, 245–254.
        """

        exp_total_rows, exp_total_cols = self.total_rows, self.total_cols
        self.total_rows, self.total_cols = self.img_rows, self.img_cols

        def residual_block(H_start, num_filter, rescale_input=False):
            """ Make a residual connection as described in the paper

            Order is ELU -> 3x3 -> Dropout -> ELU -> 3x3 -> Scale -> Add
            """
            H = self.activation_cls()(H_start)
            H = Convolution2D(num_filter, (3, 3),
                              padding='same')(H)
            H = Dropout(DROPOUT_RATE)(H)
            H = self.activation_cls()(H)
            H = Convolution2D(num_filter, (3, 3),
                              padding='same')(H)
            H = Lambda(lambda x: x * 0.3)(H)
            if rescale_input:
                H_start = Convolution2D(num_filter, (1, 1),
                                        padding='same')(H_start)
            H = Add()([H_start, H])
            return H

        input_net = Input(shape=(self.img_rows, self.img_cols, self.img_colors))

        # First block down 100x100
        net = Convolution2D(32, (3, 3),
                            padding='same')(input_net)
        net = net1 = residual_block(net, num_filter=32)
        net = Convolution2D(64, (3, 3),
                            padding='same')(net)

        # Second block down 50x50
        zero_pad1 = self.total_rows % 2
        self.total_rows //= 2
        self.total_cols //= 2
        net = AveragePooling2D(pool_size=(2, 2))(net)
        net = net2 = residual_block(net, num_filter=64)
        net = Convolution2D(128, (3, 3),
                            padding='same')(net)

        # Third block down 25x25
        zero_pad2 = self.total_rows % 2
        self.total_rows //= 2
        self.total_cols //= 2
        net = AveragePooling2D(pool_size=(2, 2))(net)
        net = net3 = residual_block(net, num_filter=128)
        net = Convolution2D(256, (3, 3),
                            padding='same')(net)

        # Fourth block down 12x12
        zero_pad3 = self.total_rows % 2
        self.total_rows //= 2
        self.total_cols //= 2
        net = AveragePooling2D(pool_size=(2, 2))(net)
        net = net4 = residual_block(net, num_filter=256)
        net = Convolution2D(256, (3, 3),
                            padding='same')(net)

        # Bottom block 6x6
        zero_pad4 = self.total_rows % 2
        self.total_rows //= 2
        self.total_cols //= 2
        net = AveragePooling2D(pool_size=(2, 2))(net)
        net = residual_block(net, num_filter=256, rescale_input=True)

        print('Bottom shape: {}x{}'.format(self.total_rows, self.total_cols))
        print('Padding: {},{},{},{}'.format(zero_pad1, zero_pad2, zero_pad3, zero_pad4))

        # First block up 12x12
        self.total_rows *= 2
        self.total_cols *= 2
        net = UpSampling2D(size=(2, 2),
                           data_format='channels_last')(net)
        if zero_pad4 == 1:
            self.total_rows += 1
            self.total_cols += 1
            net = ZeroPadding2D(padding=((1, 0), (1, 0)))(net)
        net = Concatenate(axis=-1)([net, net4])
        net = residual_block(net, num_filter=256, rescale_input=True)

        # Second block up 25x25
        self.total_rows = self.total_rows*2
        self.total_cols = self.total_cols*2
        net = UpSampling2D(size=(2, 2),
                           data_format='channels_last')(net)
        if zero_pad3 == 1:
            self.total_rows += 1
            self.total_cols += 1
            net = ZeroPadding2D(padding=((1, 0), (1, 0)))(net)
        net = Concatenate(axis=-1)([net, net3])
        net = residual_block(net, num_filter=128, rescale_input=True)

        # Third block up 50x50
        self.total_rows *= 2
        self.total_cols *= 2
        net = UpSampling2D(size=(2, 2),
                           data_format='channels_last')(net)
        if zero_pad2 == 1:
            self.total_rows += 1
            self.total_cols += 1
            net = ZeroPadding2D(padding=((1, 0), (1, 0)))(net)
        net = Concatenate(axis=-1)([net, net2])
        net = residual_block(net, num_filter=64, rescale_input=True)

        # Fourth block up 100x100
        self.total_rows *= 2
        self.total_cols *= 2
        net = UpSampling2D(size=(2, 2),
                           data_format='channels_last')(net)
        if zero_pad1 == 1:
            self.total_rows += 1
            self.total_cols += 1
            net = ZeroPadding2D(padding=((1, 0), (1, 0)))(net)
        net = Concatenate(axis=-1)([net, net1])
        net = residual_block(net, num_filter=32, rescale_input=True)
        net = Convolution2D(1, (3, 3),
                            padding='same')(net)
        print('Got final size: {}x{}'.format(self.total_rows, self.total_cols))
        print('Expected size:  {}x{}'.format(exp_total_rows, exp_total_cols))
        assert (self.total_rows, self.total_cols) == (exp_total_rows, exp_total_cols)

        detector = Model(input_net, net)
        detector.compile(loss=loss_function, optimizer=self.opt)
        detector.summary()

        self.detector = detector

    # Additional nets with different FOV parameters
    make_detector_fcrn_a_wide = make_detector_fcrn_a

    make_detector_fcrn_b_wide = make_detector_fcrn_b

    make_detector_residual_unet_wide = make_detector_residual_unet

    # Target data formatting

    def reshape_y_data_counts(self, y_real):
        """ Convert the input masks into a score

        :param y_real:
            The batch_size x img_rows x img_cols x 1 array of masks
        :returns:
            A batch_size x 1 array of scores for each mask
        """
        y_real = np.reshape(y_real, (-1, self.img_rows*self.img_cols))
        y = np.sum(y_real, axis=1)[:, np.newaxis] / self.img_rows / self.img_cols
        return y

    def reshape_y_data_points(self, y_real):
        """ Convert the input masks into a score

        :param y_real:
            The batch_size x img_rows x img_cols x 1 array of masks
        :returns:
            A batch_size x 1 array of scores for each mask
        """
        cx = self.img_rows//2
        cy = self.img_cols//2
        sampling = (self.peak_sharpness / self.img_rows,
                    self.peak_sharpness / self.img_cols)

        y = []
        for arr in y_real:
            arr = arr[:, :, 0] == 0
            assert arr.shape == (self.img_rows, self.img_cols)
            if np.all(arr):
                y.append(0.0)
            else:
                dist = distance_transform_edt(arr, return_distances=True,
                                              sampling=sampling)
                y.append(1.0 - clamp(dist[cx, cy], 0.0, 1.0))
        y = np.array(y)[:, np.newaxis]
        return y

    def reshape_y_data_peaks(self, y_real):
        """ Convert the peak data in y_real into a score

        :param y_real:
            The batch_size x img_rows x img_cols x 1 array of masks
        :returns:
            A batch_size x 1 array of scores for each mask
        """
        cx = self.img_rows//2
        cy = self.img_cols//2
        cx_st = cx - self.total_rows//2
        cx_ed = cx_st + self.total_rows

        cy_st = cy - self.total_cols//2
        cy_ed = cy_st + self.total_cols

        print(f'Got {cx_ed - cx_st}x{cy_ed - cy_st}')
        print(f'Expected {self.total_rows}x{self.total_cols}')

        y = []
        for arr in y_real:
            y.append(arr[cx_st:cx_ed, cy_st:cy_ed, 0] / 255)
        y = np.stack(y, axis=0)[..., np.newaxis]
        print(f'Final shape: {y.shape}')
        return y

    # def visualize_saliency(self, image_file, outdir):
    #     """ Dump the saliency maps from the layer
    #
    #     :param Path image_file:
    #         The path to an image file in the training data
    #     :param Path outdir:
    #         The directory to save the plots to
    #     """
    #     if image_file in self.x_train.files:
    #         img = self.x_train.load_file(image_file)
    #     else:
    #         img = self.x_test.load_file(image_file)
    #
    #     img = img[104:136, 157:189, :]
    #     img = img[np.newaxis, ...]
    #
    #     assert img.shape == (1, IMG_ROWS, IMG_COLS, IMG_COLORS)
    #
    #     if outdir.is_dir():
    #         shutil.rmtree(str(outdir))
    #     outdir.mkdir(exist_ok=True, parents=True)
    #
    #     # Batch norms break the layer visualization
    #     self.make_detector(detector='countception', skip_batch_norm=True)
    #     model = self.detector
    #
    #     def get_filter_output(layer_idx, filter_indices):
    #         # Stolen from their garbage code
    #         # `ActivationMaximization` loss reduces as outputs get large, hence negative gradients indicate the direction
    #         # for increasing activations. Multiply with -1 so that positive gradients indicate increase instead.
    #         wrt_tensor = None
    #         grad_modifier = 'absolute'
    #         losses = [
    #             (ActivationMaximization(model.layers[layer_idx], filter_indices), -1)
    #         ]
    #         opt = Optimizer(model.input, losses, wrt_tensor=wrt_tensor, norm_grads=False)
    #         grads = opt.minimize(seed_input=img, max_iter=1, grad_modifier=grad_modifier, verbose=False)[1]
    #         channel_idx = -1
    #         return np.max(grads, axis=channel_idx)
    #
    #     target_layers = [i for i, l in enumerate(model.layers) if isinstance(l, LeakyReLU)]
    #     for layer_number in target_layers:
    #         num_filters = model.layers[layer_number].output_shape[-1]
    #         print('Layer {} has {} filters'.format(layer_number, num_filters))
    #
    #         grad_targets = [(None, 'layer{:02d}_combined'.format(layer_number))]
    #
    #         for filter_index, title in grad_targets:
    #             grads = get_filter_output(layer_number, filter_index)
    #             assert grads.shape == (1, IMG_ROWS, IMG_COLS)
    #
    #             with set_plot_style(PLOT_STYLE) as style:
    #                 fig, (ax1, ax2) = plt.subplots(1, 2)
    #
    #                 ax1.imshow(img[0, :, :, 0], cmap='viridis')
    #                 ax2.imshow(grads[0, :, :], cmap='inferno')
    #
    #                 ax1.set_xticks([])
    #                 ax1.set_yticks([])
    #
    #                 ax2.set_xticks([])
    #                 ax2.set_yticks([])
    #
    #                 fig.suptitle(title)
    #                 style.savefig(str(outdir / '{}.png'.format(title)),
    #                               transparent=True)
    #                 plt.close()

    def visualize_weights(self, image_file, outdir):
        """ Dump weights from a layer on an example image

        :param Path image_file:
            The path to an image file in the training data
        :param Path outdir:
            The directory to save the plots to
        """
        if image_file in self.x_train.files:
            img = self.x_train.load_file(image_file)
        else:
            img = self.x_test.load_file(image_file)

        img = img[104:136, 157:189, :]

        if outdir.is_dir():
            shutil.rmtree(str(outdir))
        outdir.mkdir(exist_ok=True, parents=True)

        orig_img = img

        # Batch norms break the layer visualization
        self.make_detector(detector='countception', skip_batch_norm=True)
        model = self.detector

        def get_layer_outputs(image):
            '''This function extracts the numerical output of each layer.'''
            outputs = [layer.output for layer in model.layers]
            comp_graph = [K.function([model.input] + [K.learning_phase()], [output]) for output in outputs]

            # Feeding the image
            layer_outputs_list = [op([[image]]) for op in comp_graph]

            layer_outputs = []
            for layer_output in layer_outputs_list:
                layer_outputs.append(layer_output[0][0])

            return layer_outputs

        def plot_layer_outputs(image, layer_number):
            '''This function handels plotting of the layers'''
            layer_outputs = get_layer_outputs(image)

            x_max = layer_outputs[layer_number].shape[0]
            y_max = layer_outputs[layer_number].shape[1]
            n = layer_outputs[layer_number].shape[2]

            L = []
            for i in range(n):
                L.append(np.zeros((x_max, y_max)))

            for i in range(n):
                for x in range(x_max):
                    for y in range(y_max):
                        L[i][x][y] = layer_outputs[layer_number][x][y][i]

            for i, img in enumerate(L):
                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(orig_img[..., 0], cmap='viridis')

                ax1.set_xticks([])
                ax1.set_yticks([])

                ax2.imshow(img, interpolation='nearest', cmap='inferno')
                ax2.set_xticks([])
                ax2.set_yticks([])

                outfile = outdir / 'layer{:03d}'.format(layer_number) / 'filter{:03d}.png'.format(i)
                outfile.parent.mkdir(exist_ok=True, parents=True)
                fig.savefig(str(outfile), transparent=True)
                plt.close()

        target_layers = [i for i, l in enumerate(model.layers) if isinstance(l, LeakyReLU)]
        for layer_number in target_layers:
            plot_layer_outputs(img, layer_number)

    def plot_response(self,
                      image_file: Optional[pathlib.Path] = None,
                      show: bool = True,
                      plotfile: Optional[pathlib.Path] = None,
                      pointfile: Optional[pathlib.Path] = None,
                      response_file: Optional[pathlib.Path] = None,
                      data_resize: int = DATA_RESIZE,
                      composite_stride: int = COMPOSITE_STRIDE,
                      composite_mode: str = COMPOSITE_MODE,
                      composite_transforms: str = COMPOSITE_TRANSFORMS,
                      composite_batch_size: int = COMPOSITE_BATCH_SIZE,
                      timing_log_fp=None):
        """ Plot the response over the whole image

        :param Path image_file:
            If not None, the image_file to track, otherwise a random training image
        :param bool show:
            If True, plot the result in matplotlib
        :param Path plotfile:
            If not None, save the plot to a file
        :param Path pointfile:
            The path to save the resulting peak points to
        :param Path response_file:
            The path to save the response detections to
        :param int data_resize:
            How much to downsample the image by
        :param int composite_stride:
            Spacing between strides in the composite image
        :param str composite_mode:
            How to composite the image (either peak or mean)
        :param str composite_transforms:
            Transforms to merge (ether none or rotations)
        """
        print(f'Got {self.total_rows}x{self.total_cols} output shape')

        if image_file is None:
            image_file, img, mask = self.x_test.load_random_image()
        else:
            image_file = image_file.resolve()
            img = self.x_train.load_file(image_file)
            # mask = self.x_train.load_mask(image_file, img)

        print('Original shape: {}'.format(img.shape))

        if data_resize != 1:
            print('Reducing image by: {}'.format(data_resize))
            if img.ndim == 2:
                scales = (data_resize, data_resize)
            else:
                scales = (data_resize, data_resize, 1)
            img = block_reduce(img, scales)
            print('Reduced shape:  {}'.format(img.shape))

        img_min = np.min(img)
        img_max = np.max(img)
        print('Original range: {} to {}'.format(img_min, img_max))

        if data_resize != 1:
            print('Rescaling the image')
            img = img / data_resize**2
            print('New range: {} to {}'.format(np.min(img), np.max(img)))

        # Composite the data
        t0 = time.monotonic()
        response = predict_with_steps(
            img=img,
            detector=self.detector,
            img_shape=(self.img_rows, self.img_cols),
            response_shape=(self.total_rows, self.total_cols),
        )
        min_response = np.nanmin(response)
        response[np.isnan(response)] = min_response
        detect_time = time.monotonic() - t0
        print(f'Finished composite in {detect_time:0.4f} secs')

        if self.detection_erosion > 0:
            de = self.detection_erosion
            response[:de, :] = min_response
            response[-de:, :] = min_response
            response[:, :de] = min_response
            response[:, -de:] = min_response

        if response_file is not None:
            print(f'Saving response: {response_file}')
            print(f'Minimum response: {np.min(response)}')
            print(f'Maximum response: {np.max(response)}')
            save_image(response_file, response,
                       cmin=0.0, cmax=1.0)

        t0 = time.monotonic()
        peaks = peak_local_max(
            response + np.random.rand(*response.shape)*1e-5,
            min_distance=int(np.ceil(self.peak_distance)),
            threshold_abs=self.detection_threshold,
            exclude_border=0)

        # Dump the points to a file
        rows, cols = img.shape[:2]

        assert peaks.ndim == 2
        assert peaks.shape[0] > 0
        assert peaks.shape[1] == 2
        if pointfile is not None:
            norm_px = [(px - self.zero_padding) / (cols - 2*self.zero_padding)
                       for px in peaks[:, 1]]
            norm_py = [(1.0 - (py - self.zero_padding) / (rows - 2*self.zero_padding))
                       for py in peaks[:, 0]]
            peak_values = response[peaks[:, 0], peaks[:, 1]]
            save_point_csvfile(pointfile, norm_px, norm_py, peak_values,
                               xlim=(0, 1), ylim=(0, 1))
        peak_time = time.monotonic() - t0
        print(f'Peak detection took {peak_time:0.4f} secs')

        # Make a detector plot
        if plotfile is not None or show:
            plot_detector_response(img, response, peaks=peaks, plotfile=plotfile, show=show)

        # Log for timing purposes
        if timing_log_fp is not None:
            timing_log_fp.write(json.dumps({
                'detector_name': self.detector_name,
                'image_file': str(image_file),
                'detect_time': detect_time,
                'peak_time': peak_time,
            }) + '\n')
            timing_log_fp.flush()


# Functions


def plot_detector_response(img: np.ndarray,
                           response: np.ndarray,
                           peaks: Optional[np.ndarray] = None,
                           plotfile: Optional[pathlib.Path] = None,
                           show: bool = False,
                           response_min: float = RESPONSE_MIN,
                           response_max: float = RESPONSE_MAX,
                           plot_style: str = PLOT_STYLE):
    """ Plot the response for the detector window

    :param ndarray img:
        The 1 x rows x cols x 1 input image
    :param ndarray response:
        The 1 x rows x cols x 1 detector response image
    :param list peaks:
        If not None, a list of x, y positions for the peaks of the detector
    :param Path plotfile:
        If not None, the path to save the plot to
    :param bool show:
        If True, show the plot
    :param float response_min:
        The minimum value for the response image
    :param float response_max:
        The maximum value for the response image
    """

    if peaks is None:
        peaks = []

    # Support 3D images with one color channel
    img = np.squeeze(img)
    if img.ndim == 3:
        assert img.shape[2] == 3
    elif img.ndim != 2:
        raise ValueError(f'Got image that isn\'t 2D or RGB: {img.shape}')

    # Same deal with response objects
    response = np.squeeze(response)
    if response.ndim != 2:
        raise ValueError(f'Got response that isn\'t 2D: {response.shape}')

    response[response < response_min] = response_min
    response[response > response_max] = response_max

    rows, cols = img.shape[:2]
    axis_limits = [0, cols, rows, 0]

    # Plot the output
    with set_plot_style(plot_style) as style:
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))

        axes[0].imshow(img, cmap='viridis')
        for peak in peaks:
            axes[0].plot(peak[1], peak[0], 'or')
        axes[0].axis(axis_limits)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('Image')

        im = axes[1].imshow(response, cmap='inferno')

        axes[1].axis(axis_limits)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_title('Detection Composite')
        add_colorbar(axes[1], im)

        plt.tight_layout()

        if plotfile is not None:
            print(f'Saving plot to {plotfile}...')
            plotfile.parent.mkdir(exist_ok=True, parents=True)
            style.savefig(str(plotfile), facecolor='k', edgecolor='k', transparent=True)

        if show:
            plt.show()
        plt.close()

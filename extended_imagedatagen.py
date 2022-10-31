"""real-time data augmentation on image data with more extensions
original implementation from Keras version: 2.1.3

Modified by: Muhammad Shehzad Hanif (14/2/2018)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial
from abc import abstractmethod

from tensorflow.python.keras import backend as K   ###

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS

def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
        scale: Whether to rescale image values
            to be within [0, 255].

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def im_resize(x, scale, data_format):
    resample = _PIL_INTERPOLATION_METHODS['bilinear']
    width_height_tuple = (int(x.shape[1]*scale), int(x.shape[0]*scale))
    img = array_to_img(x, data_format=data_format)
    img = img.resize(width_height_tuple, resample)
    x = img_to_array(img, data_format=data_format)
    return x

class SimNetImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.
    """
    def __init__(self,
                 data_file_name,
                 match_file_name,
                 network_type,
                 preprocessing=False,
                 rotation_range=None,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False):

        #load data from file and get matches and non-mathces indides
        print('Loading data file %s\nUsing match file %s' %(data_file_name, match_file_name))
        self.data = np.load(file=open(data_file_name, 'rb'))[()]
        match_data = self.data['match_data']
        self.similar_idx = match_data[match_file_name][0]
        self.nonsimilar_idx = match_data[match_file_name][1]

        # 2ch, 2ch2stream or siam
        if network_type == 'siam_l2': #treat siam_l2 and siam in batch generation
            self.network_type = 'siam'
        else:
            self.network_type = network_type

        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.preprocessing = preprocessing
        self.channel_axis = 3
        self.row_axis = 1
        self.col_axis = 2

    def get_total_number_of_samples(self):
        """ Get number of samples in the train set
        """
        return self.similar_idx.shape[0] + self.nonsimilar_idx.shape[0]

    def flow(self, batch_size=32, shuffle=True, seed=None):
        return NumpyArrayIterator(
            self,
            batch_size=batch_size // 2,  ### (Shehzad): Divide the batch size by 2 so that when batches are created, it will be equal to the batch size
            shuffle=shuffle,
            seed=seed)

    def standardize(self, x):
        if self.preprocessing:
            m = np.mean(x, (0,1))
            x -= m
            x /= 255.0

        return x


class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    # Notes

    `Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train once
     on each sample per epoch which is not the case with generators.

    # Examples

    ```python
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np
        import math

        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.

        class CIFAR10Sequence(Sequence):

            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return math.ceil(len(self.x) / self.batch_size)

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return np.array([
                    resize(imread(file_name), (200, 200))
                       for file_name in batch_x]), np.array(batch_y)
    ```
    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass


class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.

    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ExtendedImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, image_data_generator,
                 batch_size=32, shuffle=False, seed=None):

        self.image_data_generator = image_data_generator
        n = image_data_generator.similar_idx.shape[0]
        super(NumpyArrayIterator, self).__init__(n, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        #get shape of the data
        im = self.image_data_generator.data['patches'][0, :, :]
        if self.image_data_generator.network_type == '2ch' or self.image_data_generator.network_type == 'siam':
            #it is simple 2 channel network
            batch_x = np.zeros(tuple([len(index_array)*2] + list(im.shape) + [2]), dtype=K.floatx())
            batch_y = np.zeros([len(index_array)*2], dtype=np.int32)
            for i, j in enumerate(index_array):
                sim_idx = self.image_data_generator.similar_idx[j, :]
                nonsim_idx = self.image_data_generator.nonsimilar_idx[j, :]
                x1_sim = self.image_data_generator.data['patches'][sim_idx[0], :, :]
                x2_sim = self.image_data_generator.data['patches'][sim_idx[1], :, :]
                x1_nonsim = self.image_data_generator.data['patches'][nonsim_idx[0], :, :]
                x2_nonsim = self.image_data_generator.data['patches'][nonsim_idx[1], :, :]
                x1_sim = self.image_data_generator.standardize(x1_sim.astype(K.floatx()))
                x2_sim = self.image_data_generator.standardize(x2_sim.astype(K.floatx()))
                x1_nonsim = self.image_data_generator.standardize(x1_nonsim.astype(K.floatx()))
                x2_nonsim = self.image_data_generator.standardize(x2_nonsim.astype(K.floatx()))
                batch_x[2*i,:,:,0,None] = x1_sim[:,:,None]
                batch_x[2*i,:,:,1,None] = x2_sim[:,:,None]
                batch_x[(2*i)+1,:,:,0,None] = x1_nonsim[:,:,None]
                batch_x[(2*i)+1,:,:,1,None] = x2_nonsim[:,:,None]
                batch_y[2*i] = 1
                batch_y[(2*i)+1] = -1

            if self.shuffle == True:
                idx = np.random.choice(np.arange(len(index_array)*2), len(index_array)*2, replace=False)
                batch_x = batch_x[idx, :, :, :]
                batch_y = batch_y[idx]

            if self.image_data_generator.network_type == '2ch':
                return (batch_x, batch_y)
            else:
                #for siamese network
                return ([batch_x[:,:,:,0,None], batch_x[:,:,:,1,None]], batch_y)
        else:
            # it is 2 channel 2 stream network
            batch_xF = np.zeros(tuple([len(index_array) * 2] + list((im.shape[0]//2, im.shape[1]//2)) + [2]), dtype=K.floatx())
            batch_xR = np.zeros(tuple([len(index_array) * 2] + list((im.shape[0]//2, im.shape[1]//2)) + [2]), dtype=K.floatx())
            batch_y = np.zeros([len(index_array) * 2], dtype=np.int32)
            for i, j in enumerate(index_array):
                sim_idx = self.image_data_generator.similar_idx[j, :]
                nonsim_idx = self.image_data_generator.nonsimilar_idx[j, :]
                x1_sim = self.image_data_generator.data['patches'][sim_idx[0], :, :]
                x2_sim = self.image_data_generator.data['patches'][sim_idx[1], :, :]
                x1_nonsim = self.image_data_generator.data['patches'][nonsim_idx[0], :, :]
                x2_nonsim = self.image_data_generator.data['patches'][nonsim_idx[1], :, :]

                #get retina and fovea for each image
                x1_sim_retina = x1_sim[16:48, 16:48,None]
                x1_sim_fovea = im_resize(x1_sim[:,:,None], scale=0.5, data_format=K.image_data_format())
                x2_sim_retina = x2_sim[16:48, 16:48,None]
                x2_sim_fovea = im_resize(x2_sim[:,:,None], scale=0.5, data_format=K.image_data_format())
                x1_nonsim_retina = x1_nonsim[16:48, 16:48,None]
                x1_nonsim_fovea = im_resize(x1_nonsim[:,:,None], scale=0.5, data_format=K.image_data_format())
                x2_nonsim_retina = x2_nonsim[16:48, 16:48,None]
                x2_nonsim_fovea = im_resize(x2_nonsim[:,:,None], scale=0.5, data_format=K.image_data_format())

                #apply preprocessing
                x1_sim_retina = self.image_data_generator.standardize(x1_sim_retina.astype(K.floatx()))
                x2_sim_retina = self.image_data_generator.standardize(x2_sim_retina.astype(K.floatx()))
                x1_nonsim_retina = self.image_data_generator.standardize(x1_nonsim_retina.astype(K.floatx()))
                x2_nonsim_retina = self.image_data_generator.standardize(x2_nonsim_retina.astype(K.floatx()))
                x1_sim_fovea = self.image_data_generator.standardize(x1_sim_fovea.astype(K.floatx()))
                x2_sim_fovea = self.image_data_generator.standardize(x2_sim_fovea.astype(K.floatx()))
                x1_nonsim_fovea = self.image_data_generator.standardize(x1_nonsim_fovea.astype(K.floatx()))
                x2_nonsim_fovea = self.image_data_generator.standardize(x2_nonsim_fovea.astype(K.floatx()))

                batch_xR[2 * i, :, :, 0, None] = x1_sim_retina
                batch_xR[2 * i, :, :, 1, None] = x2_sim_retina
                batch_xR[(2 * i) + 1, :, :, 0, None] = x1_nonsim_retina
                batch_xR[(2 * i) + 1, :, :, 1, None] = x2_nonsim_retina
                batch_xF[2 * i, :, :, 0, None] = x1_sim_fovea
                batch_xF[2 * i, :, :, 1, None] = x2_sim_fovea
                batch_xF[(2 * i) + 1, :, :, 0, None] = x1_nonsim_fovea
                batch_xF[(2 * i) + 1, :, :, 1, None] = x2_nonsim_fovea
                batch_y[2 * i] = 1
                batch_y[(2 * i) + 1] = -1

            if self.shuffle == True:
                idx = np.random.choice(np.arange(len(index_array) * 2), len(index_array) * 2, replace=False)
                batch_xR = batch_xR[idx, :, :, :]
                batch_xF = batch_xF[idx, :, :, :]
                batch_y = batch_y[idx]
            return ([batch_xR, batch_xF], batch_y)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
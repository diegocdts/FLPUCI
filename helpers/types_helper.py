import os
from enum import Enum

from helpers.parameters_helper import SampleSelectionType

root_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), 'Data-CommunitiesIdentificationThroughFL/')


class TypeLearning(Enum):
    CEN = 'CEN'
    FED = 'FED'

    def __str__(self):
        return str(self.value)


class LatYLonXTimeIndices:

    def __init__(self, lat_y: int, lon_x: int, time: int):
        self.lat_y = lat_y
        self.lon_x = lon_x
        self.time = time


class Dataset:

    def __init__(self,
                 name: str,
                 hours_per_window: float,
                 first_window: int,
                 lat_y_min: float,
                 lat_y_max: float,
                 lon_x_min: float,
                 lon_x_max: float,
                 resolution: tuple,
                 indices: LatYLonXTimeIndices,
                 last_window: int = None,
                 is_lat_lon: bool = True,
                 paddingYX: tuple = (False, False)):
        self.name = name
        self.hours_per_window = hours_per_window
        self.first_window = first_window
        self.last_window = last_window
        self.lat_y_min = lat_y_min
        self.lat_y_max = lat_y_max
        self.lon_x_min = lon_x_min
        self.lon_x_max = lon_x_max
        self.resolution = resolution
        self.is_lat_lon = is_lat_lon
        self.paddingYX = paddingYX
        self.indices = indices
        self.epoch_size = len(str(first_window))
        self.height = None
        self.width = None

    def set_height_width(self, float_height: float, float_width: float):
        if float_height.is_integer:
            float_height = int(float_height) + 1
        if int(float_height) % 2 != 0:
            float_height += 1
        self.height = int(float_height)

        if float_width.is_integer:
            float_width = int(float_width) + 1
        if int(float_width) % 2 != 0:
            float_width += 1
        self.width = int(float_width)

        if self.paddingYX[0]:
            self.height = self.height + 2
        if self.paddingYX[1]:
            self.width = self.width + 2


class FCAEProperties:

    def __init__(self,
                 input_shape: tuple,
                 encode_layers: list,
                 encode_activation: str,
                 decode_activation: str,
                 kernel_size: tuple,
                 encode_strides: list,
                 padding: str,
                 latent_space: int,
                 learning_rate: float):
        self.input_shape = input_shape
        self.encode_layers = encode_layers
        self.decode_layers = encode_layers[::-1]
        self.encode_activation = encode_activation
        self.decode_activation = decode_activation
        self.kernel_size = kernel_size
        self.encode_strides = encode_strides
        self.decode_strides = encode_strides[::-1]
        self.padding = padding
        self.latent_space = latent_space
        self.learning_rate = learning_rate


class TrainingParameters:

    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 shuffle_buffer: int = None,
                 prefetch_buffer: int = None,
                 rounds: int = None):
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.rounds = rounds


class SampleSelectionParameters:

    def __init__(self,
                 sample_selection_type: SampleSelectionType,
                 window_size: int):
        self.sample_selection_type = sample_selection_type
        self.window_size = window_size

    def set_window_size(self, window_size):
        self.window_size = window_size

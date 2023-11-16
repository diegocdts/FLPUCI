import tensorflow as tf

from functions.files_paths_functions import get_file_path, win_space, path_exists
from helpers.files_paths_helper import Path
from helpers.types_helper import FCAEProperties, TypeLearning, TrainingParameters
from steps.step_02_sample_generation import SampleHandler
from graphics import LossesHandler
import gc


class FullConvolutionalAutoEncoder:

    def __init__(self, sample_handler: SampleHandler, properties: FCAEProperties):
        self.sample_handler = sample_handler
        self.f9_checkpoint = Path.f9_checkpoints(self.sample_handler.dataset.name, TypeLearning.CEN)
        self.properties = properties
        self.encoder = encoder_build(properties)
        self.decoder = decoder_build(properties)
        self.model = model_build(properties)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.properties.learning_rate),
                           loss=tf.keras.losses.MeanSquaredError())

    @staticmethod
    def checkpoint(path: str):
        return tf.keras.callbacks.ModelCheckpoint(filepath=get_file_path(path, 'ckpt'),
                                                  save_weights_only=True, verbose=1)

    def training(self, start_window: int, end_window: int, parameters: TrainingParameters):
        path = get_file_path(self.f9_checkpoint, win_space(start_window, end_window))
        loss_handler = LossesHandler(path, TypeLearning.CEN)

        if path_exists(path):
            loss_handler.load_cen()
            self.model.load_weights(get_file_path(path, 'ckpt'))
        else:
            training_data, training_indices_list = self.sample_handler.samples_as_list(start_window, end_window)
            testing_data, testing_indices_list = self.sample_handler.samples_as_list(end_window, end_window + 1)
            history = self.model.fit(training_data, training_data, batch_size=parameters.batch_size,
                                     epochs=parameters.epochs, verbose=1, callbacks=[self.checkpoint(path)],
                                     validation_data=(testing_data, testing_data))
            loss_handler.append_cen(history.history['loss'], history.history['val_loss'])
            loss_handler.save()
            del training_data, testing_data, loss_handler
        gc.collect()

    def encoder_prediction(self, start_window: int, end_window: int):
        samples, indices_list = self.sample_handler.samples_as_list(start_window, end_window)
        encoder = trained_encoder(self.model)
        predictions = encoder.predict(samples)
        del samples, encoder
        gc.collect()
        return predictions, indices_list


def dense_nodes_width_height(fcaep: FCAEProperties):
    width, height = fcaep.input_shape[0], fcaep.input_shape[1]
    for _ in fcaep.encode_layers:
        width = width / 2
        height = height / 2
    dense_nodes = int(width * height * fcaep.encode_layers[-1])
    return dense_nodes, int(width), int(height)


def encoder_build(fcaep: FCAEProperties):
    encoder = tf.keras.Sequential()
    encoder.add(tf.keras.layers.InputLayer(input_shape=fcaep.input_shape))

    for layer in fcaep.encode_layers:
        encoder.add(tf.keras.layers.Conv2D(layer, fcaep.kernel_size, activation=fcaep.encode_activation,
                                           strides=fcaep.strides, padding=fcaep.padding))
    encoder.add(tf.keras.layers.Flatten())
    encoder.add(tf.keras.layers.Dense(fcaep.latent_space, activation=fcaep.encode_activation))
    return encoder


def decoder_build(fcaep: FCAEProperties):
    dense_layer, width, height = dense_nodes_width_height(fcaep)

    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.InputLayer(input_shape=(fcaep.latent_space,)))

    decoder.add(tf.keras.layers.Dense(dense_layer, activation=fcaep.decode_activation))
    decoder.add(tf.keras.layers.Reshape((height, width, fcaep.decode_layers[0])))

    for layer in fcaep.decode_layers:
        decoder.add(tf.keras.layers.Conv2DTranspose(layer, fcaep.kernel_size, activation=fcaep.decode_activation,
                                                    strides=fcaep.strides, padding=fcaep.padding))
    decoder.add(tf.keras.layers.Conv2DTranspose(1, fcaep.kernel_size, activation=fcaep.decode_activation,
                                                padding=fcaep.padding))
    return decoder


def model_build(fcaep: FCAEProperties):
    encoder = encoder_build(fcaep)
    decoder = decoder_build(fcaep)
    return tf.keras.models.Model(inputs=encoder.input, outputs=decoder(encoder.outputs))


def trained_encoder(model: tf.keras.Model):
    return tf.keras.models.Model(model.input, model.layers[-2].output)

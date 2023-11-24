import asyncio
import tensorflow as tf
import tensorflow_federated as tff
import collections
import gc

from functions.console_functions import time
from functions.files_paths_functions import win_space, get_file_path, dir_create
from graphics import LossesHandler
from helpers.files_paths_helper import Path
from helpers.types_helper import Dataset, TrainingParameters, FCAEProperties, TypeLearning
from steps.step_02_sample_generation import SampleHandler
from steps.step_04_deep_learning_model import model_build, trained_encoder


class FederatedDataHandler:

    def __init__(self, dataset: Dataset, training_parameters: TrainingParameters):
        self.sample_handler = SampleHandler(dataset=dataset)
        self.training_parameters = training_parameters
        self.element_spec = self.element_spec_build()

    def preprocess(self, dataset):

        batch_size = self.training_parameters.batch_size
        if len(dataset) < batch_size:
            batch_size = len(dataset)

        def batch_format_fn(element):
            return collections.OrderedDict(x=element, y=element)

        return dataset.repeat(self.training_parameters.epochs).shuffle(self.training_parameters.shuffle_buffer).batch(
            batch_size).map(batch_format_fn).prefetch(self.training_parameters.prefetch_buffer)

    def element_spec_build(self):
        single_user_dataset = tf.data.Dataset.from_tensor_slices(self.sample_handler.random_dataset())
        preprocessed = self.preprocess(single_user_dataset)
        del single_user_dataset
        return preprocessed.element_spec

    def users_data(self, start_window: int, end_window: int):
        users_dataset_samples, indices_list = self.sample_handler.get_datasets(start_window, end_window)
        federated_dataset_samples = []

        for dataset in users_dataset_samples:
            if len(dataset) > 0:
                federated_dataset = tf.data.Dataset.from_tensor_slices(dataset)
                preprocessed = self.preprocess(federated_dataset)
                federated_dataset_samples.append(preprocessed)
        del users_dataset_samples
        return federated_dataset_samples


class FederatedFullConvolutionalAutoEncoder:

    def __init__(self, federated_data_handler: FederatedDataHandler, properties: FCAEProperties):
        self.properties = properties
        self.federated_data_handler = federated_data_handler
        self.iterative_process, self.state = self.global_model_start()
        self.evaluator = self.build_evaluator()
        self.dataset_name = federated_data_handler.sample_handler.dataset.name
        self.f9_checkpoint = Path.f9_checkpoints(self.dataset_name, TypeLearning.FED)
        self.state_manager = None

    def model_fn(self):
        keras_model = model_build(self.properties)
        return tff.learning.models.from_keras_model(
            keras_model=keras_model,
            input_spec=self.federated_data_handler.element_spec,
            loss=tf.keras.losses.MeanSquaredError()
        )

    def global_model_start(self):
        iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=self.properties.learning_rate),
            server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=self.properties.learning_rate)
        )
        # print(str(iterative_process.initialize.type_signature))
        return iterative_process, iterative_process.initialize()

    def build_evaluator(self):
        return tff.learning.build_federated_evaluation(self.model_fn)

    def model_evaluation(self, testing_data):
        return self.evaluator(self.state.global_model_weights, testing_data)

    def init_state_manager(self, path, rounds):
        dir_create(path)
        self.state_manager = tff.program.FileProgramStateManager(root_dir=path, prefix='round_', keep_total=rounds,
                                                                 keep_first=True)

    def get_next_round(self, loop):
        last_state, last_round = loop.run_until_complete(self.state_manager.load_latest(self.state))
        if last_state is not None:
            self.state = last_state
            return last_round + 1
        else:
            return 0

    def training(self, start_window: int, end_window: int):
        loop = asyncio.get_event_loop()
        rounds = self.federated_data_handler.training_parameters.rounds
        path = get_file_path(self.f9_checkpoint, win_space(start_window, end_window))
        self.init_state_manager(path, rounds)
        next_round = self.get_next_round(loop)

        loss_handler = LossesHandler(path, TypeLearning.FED)

        if next_round < rounds:

            training_data = self.federated_data_handler.users_data(start_window, end_window)
            testing_data = self.federated_data_handler.users_data(end_window, end_window + 1)

            for round_num in range(0, rounds):
                print('[{}] start: {} | end: {} | round: {}'.format(time(), start_window, end_window, round_num))
                round_iteration = self.iterative_process.next(self.state, training_data)
                self.state = round_iteration[0]
                loop.run_until_complete(self.state_manager.save(self.state, round_num))

                loss_handler.append_fed(round_iteration[1], self.model_evaluation(testing_data))
                loss_handler.save()

            del training_data, testing_data, loss_handler
        gc.collect()

    def encoder_prediction(self, start_window: int, end_window: int):
        samples, indices_list = self.federated_data_handler.sample_handler.samples_as_list(start_window, end_window)
        keras_model = model_build(self.properties)
        self.state.global_model_weights.assign_weights_to(keras_model)
        encoder = trained_encoder(keras_model)
        predictions = encoder.predict(samples)
        del samples, encoder
        gc.collect()
        return predictions, indices_list

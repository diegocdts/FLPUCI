import gc

from functions.running_functions import adjust_first_interval, get_start_window
from graphics import IntervalEvolution, StrategiesMatch
from helpers.parameters_helper import IntervalToValidate
from helpers.types_helper import FCAEProperties, TrainingParameters, Dataset, SampleSelectionParameters, TypeLearning
from instances.sample_selection_instances import ACC
from steps.step_02_sample_generation import SampleHandler
from steps.step_04_deep_learning_model import FullConvolutionalAutoEncoder
from steps.step_05_federated_learning_model import FederatedDataHandler, FederatedFullConvolutionalAutoEncoder
from steps.step_06_validation import Validation


class Centralized:

    def __init__(self, dataset: Dataset, sample_handler: SampleHandler, fcae_properties: FCAEProperties,
                 training_parameters: TrainingParameters, sample_selection: SampleSelectionParameters):
        self.parameters = training_parameters
        self.sample_selection = sample_selection
        self.validation = Validation(dataset, TypeLearning.CEN, sample_selection.sample_selection_type)
        self.interval_evolution = IntervalEvolution(self.validation.f10_results, TypeLearning.CEN,
                                                    sample_selection.sample_selection_type)
        self.strategies_match = StrategiesMatch(path=self.validation.f10_results)
        self.model = FullConvolutionalAutoEncoder(sample_handler, fcae_properties)

    def training_validation(self, first_interval: int, last_interval: int, validate_at: IntervalToValidate):
        first_interval = adjust_first_interval(first_interval)
        for end_window in range(first_interval, last_interval):
            start_window = get_start_window(end_window, self.sample_selection)

            self.model.training(start_window, end_window, self.parameters)

            validation_start = end_window + validate_at.value - 1
            encodings, indices_list = self.model.encoder_prediction(start_window=validation_start,
                                                                    end_window=validation_start + 1)
            curves_contact_time, curves_heatmap = self.validation.window_validation(encodings, indices_list,
                                                                                    validation_start)

            self.interval_evolution.append_curves(curves_contact_time, curves_heatmap)
            del encodings, curves_contact_time, curves_heatmap
            gc.collect()
        self.interval_evolution.best_k_interval(first_interval + validate_at.value - 1, is_heatmap=False)
        self.interval_evolution.best_k_interval(first_interval + validate_at.value - 1, is_heatmap=True)


class Federated:

    def __init__(self, dataset: Dataset, fcae_properties: FCAEProperties, training_parameters: TrainingParameters,
                 sample_selection: SampleSelectionParameters):
        self.federated_data_handler = FederatedDataHandler(dataset, training_parameters)
        self.sample_selection = sample_selection
        self.validation = Validation(dataset, TypeLearning.FED, sample_selection.sample_selection_type)
        self.interval_evolution = IntervalEvolution(self.validation.f10_results, TypeLearning.FED,
                                                    sample_selection.sample_selection_type)
        self.strategies_match = StrategiesMatch(path=self.validation.f10_results)
        self.model = FederatedFullConvolutionalAutoEncoder(self.federated_data_handler, fcae_properties)

    def training_validation(self, first_interval: int, last_interval: int, validate_at: IntervalToValidate):
        first_interval = adjust_first_interval(first_interval)
        for end_window in range(first_interval, last_interval):
            start_window = get_start_window(end_window, self.sample_selection)

            self.model.training(start_window, end_window)
"""
            validation_start = end_window + validate_at.value - 1
            encodings, indices_list = self.model.encoder_prediction(start_window=validation_start,
                                                                    end_window=validation_start + 1)
            curves_contact_time, curves_heatmap = self.validation.window_validation(encodings, indices_list,
                                                                                    validation_start)

            self.interval_evolution.append_curves(curves_contact_time, curves_heatmap)
            del encodings, curves_contact_time, curves_heatmap
            gc.collect()
        self.interval_evolution.best_k_interval(first_interval + validate_at.value - 1, is_heatmap=False)
        self.interval_evolution.best_k_interval(first_interval + validate_at.value - 1, is_heatmap=True)
"""

def print_info(type_learning, dataset_object, selection):
    print('\n# ------ {} - {} - {} (size: {}) ------ #'.format(type_learning, dataset_object.name,
                                                               selection.sample_selection_type, selection.window_size))


def cen_communities_identification(dataset: Dataset,
                                   sample_handler: SampleHandler,
                                   ae_properties: FCAEProperties,
                                   training_parameters: TrainingParameters,
                                   sli_selection: SampleSelectionParameters,
                                   first_interval: int, last_interval: int,
                                   validate_at: IntervalToValidate, acc_run: bool):
    print_info(TypeLearning.CEN, dataset, sli_selection)
    sli = Centralized(dataset, sample_handler, ae_properties, training_parameters, sli_selection)
    sli.training_validation(first_interval, last_interval, validate_at)

    if acc_run:
        print_info(TypeLearning.CEN, dataset, ACC)
        acc = Centralized(dataset, sample_handler, ae_properties, training_parameters, ACC)
        acc.training_validation(first_interval, last_interval, validate_at)

        print('\n# --------- {} - {} - Match --------- #'.format(TypeLearning.CEN, dataset.name))
        StrategiesMatch.match_selection_strategy_chosen_ks(dataset, TypeLearning.CEN, first_interval + 1, last_interval)
        StrategiesMatch.ssim_match_selection_strategy_chosen_ks(
            dataset, TypeLearning.CEN, first_interval + 1, last_interval)


def fed_communities_identification(dataset: Dataset,
                                   ae_properties: FCAEProperties,
                                   training_parameters: TrainingParameters,
                                   sli_selection: SampleSelectionParameters,
                                   first_interval: int, last_interval: int,
                                   validate_at: IntervalToValidate, acc_run: bool):
    print_info(TypeLearning.FED, dataset, sli_selection)
    sli = Federated(dataset, ae_properties, training_parameters, sli_selection)
    sli.training_validation(first_interval, last_interval, validate_at)
"""
    if acc_run:
        print_info(TypeLearning.FED, dataset, ACC)
        acc = Federated(dataset, ae_properties, training_parameters, ACC)
        acc.training_validation(first_interval, last_interval, validate_at)

        print('\n# --------- {} - {} - Match --------- #'.format(TypeLearning.FED, dataset.name))
        StrategiesMatch.match_selection_strategy_chosen_ks(dataset, TypeLearning.FED, first_interval + 1, last_interval)
        StrategiesMatch.ssim_match_selection_strategy_chosen_ks(
            dataset, TypeLearning.FED, first_interval + 1, last_interval)
"""
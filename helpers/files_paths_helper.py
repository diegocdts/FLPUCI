import os
from enum import Enum

from helpers.parameters_helper import SampleSelectionType
from helpers.types_helper import TypeLearning

root_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir), 'Data-CommunitiesIdentificationThroughFL/')


def dir_exists_create(dir_name: str):
    path = os.path.join(root_dir, '{}'.format(dir_name))
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class ExportedFilesName(Enum):
    TRAINING_LOSS = 'training losses.csv'
    TESTING_LOSS = 'testing losses.csv'
    LOSSES_CURVE = 'losses curves.png'
    METRICS = '{} validation.png'
    CONTACT_TIME = 'contact time validation.png'
    CONTACT_TIME_EVOLUTION_PNG = '{} {} contact time.png'
    SSIM_EVOLUTION_PNG = '{} {} ssim.png'
    BEST_K_CSV = 'best k configuration per interval.csv'
    CONTACT_TIME_CSV = 'curves contact time.csv'
    HEATMAP_CSV = 'curves heatmap.csv'
    FIXED_K_PNG = 'fixed k configuration per interval.png'
    FIXED_K_CSV = 'fixed k configuration per interval.csv'
    K_FIXATION_GAINS = 'k fixation gains.csv'
    BEST_K_MATCH = 'best k match.png'
    LOSSES_EVOLUTION = 'LOSSES_EVOLUTION_{}.png'

    @staticmethod
    def interval_match(interval):
        return 'interval {} match.png'.format(interval)

    @staticmethod
    def ssim_interval_match(interval):
        return 'ssim interval {} match.png'.format(interval)

    @staticmethod
    def tsne_window(path: str, start_window: int, end_window: int, validation_start: int):
        path = dir_exists_create('{}/win_{}_{}/'.format(path, start_window, end_window))
        csv = '{}/prediction_{}.{}'.format(path, validation_start, 'csv')
        png = '{}/prediction_{}.{}'.format(path, validation_start, 'png')
        return csv, png

    def __str__(self):
        return str(self.value).lower()


class Path:
    @staticmethod
    def f1_raw_data(dataset: str):
        return dir_exists_create('{}/f1_raw_data'.format(dataset))

    @staticmethod
    def f2_data(dataset: str):
        return dir_exists_create('{}/f2_data/'.format(dataset))

    @staticmethod
    def f3_dm(dataset: str):
        return dir_exists_create('{}/f3_logit/'.format(dataset))

    @staticmethod
    def f4_entry_exit(dataset: str):
        return dir_exists_create('{}/f4_entry_exit/'.format(dataset))

    @staticmethod
    def f5_win_entry_exit(dataset: str):
        return dir_exists_create('{}/f5_win_entry_exit/'.format(dataset))

    @staticmethod
    def f6_contact_time(dataset: str):
        return dir_exists_create('{}/f6_contact_time/'.format(dataset))

    @staticmethod
    def f7_metrics(dataset: str):
        return dir_exists_create('{}/f7_metrics/'.format(dataset))

    @staticmethod
    def f8_results(dataset: str, type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        return dir_exists_create('{}/f8_results/{}/{}/'.format(dataset, type_learning.value, sample_selection_type))

    @staticmethod
    def f8_results_match(dataset: str, type_learning: TypeLearning):
        return dir_exists_create('{}/f8_results/{}/strategies_match/'.format(dataset, type_learning.value))

    @staticmethod
    def f9_checkpoints(dataset: str, type_learning: TypeLearning):
        return dir_exists_create('{}/f9_checkpoints/{}/'.format(dataset, type_learning.value))

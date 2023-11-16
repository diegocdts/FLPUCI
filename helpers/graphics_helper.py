from enum import Enum

from helpers.parameters_helper import SampleSelectionType
from helpers.types_helper import TypeLearning


class GraphicLabels(Enum):
    EPOCHS = 'epochs'
    ROUNDS = 'rounds'
    LOSS = 'loss'
    CONTACT_TIME_SEC = 'avg contact time (s)'
    NUM_CLUSTERS = 'number of clusters'
    INTERVAL_INDICES = 'interval indices'
    AVG_CONTACT_TIME_SEC = 'avg contact time (s)'
    K_VALUES = 'k values'

    @staticmethod
    def metric(metric: str):
        return 'normalized {}'.format(metric).upper()

    def __str__(self):
        return str(self.value).upper()


class GraphicLegends(Enum):
    TRAINING_LOSS = 'training loss'
    TESTING_LOSS = 'testing loss'
    MAX_CONTACT_TIME = 'max contact time'

    def __str__(self):
        return str(self.value).capitalize()


class GraphProps(Enum):
    FIG_SIZE_REGULAR = (5.5, 6.5)
    FIG_SIZE_LOSS = (5.5, 3)
    FIG_SIZE_CONTACT_TIME = (5.5, 5)
    FIG_SIZE_HEATMAP = (5, 4.5)
    FIG_SIZE_STRATEGY_MATCH = (5.5, 5)
    FIG_SIZE_EVOLUTION = (8.5, 5)
    FIG_SIZE_HEIGHER = (5.2, 6.8)
    FONT_SIZE_LABEL = 17
    FONT_SIZE_TICKS = 16
    FONT_SIZE_LEGEND = 13
    FONT_SIZE_SUPTITLE = 18
    BORDERPAD = 1
    LABELSPACING = 1
    LEGEND_MS = 1
    LEGEND_LOC_BETTER = 0
    LEGEND_LOC_UP_CENTER = 9
    EXTRA_LOC = 4
    EXTRA_FONT_SIZE = 10
    CAP_SIZE = 10
    CAP_THICK = 4
    MS = 10
    MARKER = ""
    LINEWIDTH_2_5 = 2.5
    LINEWIDTH_2 = 2
    HANDLETEXTPAD = -2.0


class GraphicSupTitle:
    @staticmethod
    def metrics(interval: int, type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        return 'Image metrics at interval {} ' \
               '({}|{})'.format(interval, type_learning, sample_selection_type)

    @staticmethod
    def contact_time(interval: int, type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        return 'Contact time at interval {} ' \
               '({}|{})'.format(interval, type_learning, sample_selection_type)

    @staticmethod
    def contact_time_evolution(type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        return 'Contact time evolution ({}|{})'.format(type_learning, sample_selection_type)

    @staticmethod
    def ssim_evolution(type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        return 'SSIM evolution ({}|{})'.format(type_learning, sample_selection_type)

    @staticmethod
    def fixed_k(type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        return 'Fixed k configuration per interval ({}|{})'.format(type_learning, sample_selection_type)

    @staticmethod
    def interval(interval: int):
        return 'interval {}'.format(interval).capitalize()

    @staticmethod
    def best_k_match(type_learning: TypeLearning):
        return 'Best k configuration of each sample selection approach ({})'.format(type_learning)

    @staticmethod
    def k_approach(type_learning: TypeLearning):
        return 'Chosen K comparison ({})'.format(type_learning)

    @staticmethod
    def loss_curve(start_window: int, end_window: int, type_learning: TypeLearning):
        return 'Training window [{}, {}) ({})'.format(start_window, end_window, type_learning)

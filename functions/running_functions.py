import numpy as np
import pandas as pd

from functions.files_paths_functions import get_file_path, win_txt
from helpers.parameters_helper import SampleSelectionType
from helpers.types_helper import SampleSelectionParameters


def adjust_first_interval(first_interval: int):
    if first_interval <= 0:
        first_interval = 1
    return first_interval


def get_start_window(end_window: int, sample_selection: SampleSelectionParameters):
    if sample_selection.sample_selection_type == SampleSelectionType.ACC:
        start_window = 0
    else:
        if end_window - sample_selection.window_size >= 0:
            start_window = end_window - sample_selection.window_size
        else:
            start_window = 0
    return start_window


def existing_contact_time(path: str, window: int):
    each_contact_time = pd.read_csv(get_file_path(path, win_txt(window))).to_numpy()
    non_zero = np.count_nonzero(each_contact_time)
    return int(non_zero / 2)



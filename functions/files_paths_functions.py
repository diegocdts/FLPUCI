import os

from helpers.parameters_helper import ImageMetric


def path_exists(path: str):
    return os.path.exists(path)


def dir_create(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def sorted_files(dir_path: str):
    return sorted(os.listdir(dir_path))


def get_file_path(dir_path: str, file_name: str):
    return os.path.join(dir_path, file_name)


def win_space(start_window, end_window):
    return 'win_{}_{}'.format(start_window, end_window)


def win_txt(window: int):
    return 'win_{}.txt'.format(window)


def metric_win_txt(metric: ImageMetric, window: int):
    return '{}_win_{}.txt'.format(metric.value, window)


def win_dir(window: int):
    return 'win_{}/'.format(window)

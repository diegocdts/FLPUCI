import sys

import numpy as np
import pandas as pd
import utm

from helpers.files_paths_helper import Path
from helpers.types_helper import Dataset
from helpers.parameters_helper import TimeUnits
from functions.files_paths_functions import sorted_files, get_file_path


def get_lines_and_splitter(file_path: str):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        splitter = ' '
        if ',' in lines[0]:
            splitter = ','
        first_time = int(lines[0].split(splitter)[3])
        last_time = int(lines[-1].split(splitter)[3])
        if first_time > last_time:
            lines = lines[::-1]
        return lines, splitter


class CleaningData:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.f1_raw_data = Path.f1_raw_data(dataset.name)
        self.f2_data = Path.f2_data(dataset.name)
        self.header = 'win,x,y,time\n'

    def line_split(self, line: str, splitter: str):
        split = line.split(splitter)
        lon_x = float(split[self.dataset.indices.lon_x])
        lat_y = float(split[self.dataset.indices.lat_y])
        time = split[self.dataset.indices.time].replace(' ', '').replace('\n', '')
        time = int(time)
        return lon_x, lat_y, time

    def check_last_window(self, time: int):
        return (self.dataset.last_window is not None and time <= self.dataset.last_window) or \
               self.dataset.last_window is None

    def convert_to_utm(self, lat_y, lon_x):
        if self.dataset.is_lat_lon:
            yx = utm.from_latlon(lat_y, lon_x)
            lat_y = yx[0]
            lon_x = yx[1]
        return lat_y, lon_x

    def windows_by_node(self):
        size_f1 = len(sorted_files(self.f1_raw_data))
        size_f2 = len(sorted_files(self.f2_data))
        if size_f1 != size_f2:
            window_size = self.dataset.hours_per_window * TimeUnits(self.dataset.epoch_size).HOUR

            for file_name in sorted_files(self.f1_raw_data):
                file_path = get_file_path(self.f1_raw_data, file_name)
                lines, splitter = get_lines_and_splitter(file_path)

                output_file_path = get_file_path(self.f2_data, file_name)
                with open(output_file_path, 'a') as file:
                    file.write(self.header)

                    next_window = self.dataset.first_window
                    window_index = 0

                    for line in lines:
                        lon_x, lat_y, time = self.line_split(line, splitter)

                        if self.dataset.first_window <= time and self.check_last_window(time):

                            while next_window + window_size < time:
                                next_window = next_window + window_size
                                window_index += 1

                            if self.dataset.lat_y_min <= lat_y <= self.dataset.lat_y_max and \
                                    self.dataset.lon_x_min <= lon_x <= self.dataset.lon_x_max:
                                lat_y, lon_x = self.convert_to_utm(lat_y, lon_x)
                                file.write('{},{},{},{}\n'.format(window_index, lon_x, lat_y, time))
                df = pd.read_csv(output_file_path).sort_values(by='time')
                df.to_csv(output_file_path, index=False)


def xy_adjustment(df: pd.DataFrame):
    x_min = df['x'].min()
    df['x'] -= x_min
    y_min = df['y'].min()
    df['y'] -= y_min
    df['x'] = df['x'].round(decimals=2)
    df['y'] = df['y'].round(decimals=2)
    return df


def logit(cells_list):
    epsilon = 1e-15
    arr_clipped = np.clip(cells_list, epsilon, 1 - epsilon)
    arr_logit = np.log(arr_clipped / (1 - arr_clipped))
    return arr_logit


def normalize_row(cell_stay_time: np.array, time_in_trace: int):
    time_in_trace = max(1, time_in_trace)
    cells_stay_time_logit = logit(cell_stay_time / time_in_trace)
    cells_stay_time_str = ', '.join(['{:.3f}'.format(item) for item in cells_stay_time_logit]) + '\n'
    return cells_stay_time_str


class DisplacementMatrix:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.f2_data = Path.f2_data(dataset.name)
        self.f3_dm = Path.f3_dm(dataset.name)
        self.max_window = self.get_max_window()
        print(self.dataset.width, self.dataset.height)
        self.columns = np.arange(0, self.dataset.width * self.dataset.height, 1)

    def get_max_window(self):
        max_window = 0
        min_x, min_y = sys.maxsize, sys.maxsize
        max_x, max_y = 0, 0
        for file_name in sorted_files(self.f2_data):
            file_path = get_file_path(self.f2_data, file_name)
            df = pd.read_csv(file_path)
            if df.win.max() > max_window:
                max_window = df.win.max()
            if df.x.min() < min_x:
                min_x = df.x.min()
            if df.y.min() < min_y:
                min_y = df.y.min()
            if df.x.max() > max_x:
                max_x = df.x.max()
            if df.y.max() > max_y:
                max_y = df.y.max()
        height = round((max_y - min_y) / self.dataset.resolution)
        width = round((max_x - min_x) / self.dataset.resolution)
        self.dataset.set_height_width(height, width)
        return max_window

    def fill_matrix(self, df_window: pd.DataFrame, output_file_path: str):
        cell_stay_time = np.zeros(len(self.columns))
        time_in_trace = 0

        if len(df_window) > 0:
            min_in_trace = max_in_trace = previous_time = df_window.time[0] - 1

            for index, row in df_window.iterrows():
                if row.time > max_in_trace:
                    max_in_trace = row.time

                # cell position calculation
                x_index = int(row.x / self.dataset.resolution)
                y_index = int(row.y / self.dataset.resolution)
                cell = (x_index * self.dataset.height) + y_index

                # time a node spends in a cell
                delta_time = row.time - previous_time
                new_time = cell_stay_time[cell] + delta_time
                cell_stay_time[cell] = new_time
                previous_time = row.time

            time_in_trace = max_in_trace - min_in_trace
        # appends feature_row at the matrix
        new_row = normalize_row(cell_stay_time, time_in_trace)
        output_file = open(output_file_path, 'a')
        output_file.write(new_row)

    def generate(self):
        size_f2 = len(sorted_files(self.f2_data))
        size_f3 = len(sorted_files(self.f3_dm))
        if size_f2 != size_f3:
            for file_name in sorted_files(self.f2_data):
                file_path = get_file_path(self.f2_data, file_name)
                output_file_path = get_file_path(self.f3_dm, file_name)
                df = pd.read_csv(file_path)
                df = xy_adjustment(df)
                matrix = pd.DataFrame(columns=self.columns)
                matrix.to_csv(output_file_path, index=False)
                for window in range(0, self.max_window + 1):
                    df_window = df[df.win == window]
                    copy = df_window.copy().reset_index()
                    self.fill_matrix(copy, output_file_path)


def pre_processing(dataset: Dataset):
    data = CleaningData(dataset)
    data.windows_by_node()

    data = DisplacementMatrix(dataset)
    data.generate()

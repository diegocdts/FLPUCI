import pandas as pd
import numpy as np

from functions.array_list_functions import squeeze
from functions.files_paths_functions import get_file_path, sorted_files, win_txt, path_exists, metric_win_txt
from helpers.files_paths_helper import Path
from helpers.parameters_helper import ImageMetric
from helpers.types_helper import Dataset
from steps.step_02_sample_generation import SampleHandler
from skimage.metrics import mean_squared_error as f_mse, structural_similarity as f_ssim
from sklearn.metrics.cluster import adjusted_rand_score as f_ari


class BaselineComputation:

    def __init__(self, dataset: Dataset):
        self.f2_data = Path.f2_data(dataset.name)
        self.f3_dm = Path.f3_dm(dataset.name)
        self.f4_entry_exit = Path.f4_entry_exit(dataset.name)
        self.f5_win_entry_exit = Path.f5_win_entry_exit(dataset.name)
        self.f6_contact_time = Path.f6_contact_time(dataset.name)
        self.f7_metrics = Path.f7_metrics(dataset.name)
        self.dataset_name = dataset.name
        self.resolution = dataset.resolution
        self.sample_handler = SampleHandler(dataset)
        self.width = dataset.width
        self.height = dataset.height
        self.start_window, self.end_window = self.dataset_start_end_window()

    def dataset_start_end_window(self):
        root = Path.f3_dm(self.dataset_name)
        path = get_file_path(root, sorted_files(root)[0])
        df = pd.read_csv(path)
        start_window = 0
        end_window = len(df)
        return start_window, end_window

    def cell_entry_exit(self):
        for file_name in sorted_files(self.f2_data):
            file_path = get_file_path(self.f2_data, file_name)
            output_file_path = get_file_path(self.f4_entry_exit, file_name)
            with open(file_path) as input_file:
                file_lines = input_file.readlines()[1:]
                first_line = file_lines[0].split(',')
                win = int(first_line[0])
                x, y = float(first_line[1]), float(first_line[2])
                x_i = int(x / self.resolution)
                y_i = int(y / self.resolution)
                current_cell = (x_i * self.height) + y_i
                entry_cell, exit_cell = int(first_line[3]), int(first_line[3])
                new_lines = "win,cell,entry,exit\n"
                for line in file_lines:
                    split = line.split(',')
                    window, x, y, time = int(split[0]), float(split[1]), float(split[2]), int(split[3])
                    x_index = int(x / self.resolution)
                    y_index = int(y / self.resolution)
                    cell = (x_index * self.height) + y_index
                    if cell != current_cell or win != window:
                        new_lines += "{},{},{},{}\n".format(win, current_cell, entry_cell, exit_cell)
                        win = window
                        current_cell = cell
                        entry_cell = time
                    exit_cell = time
                new_lines += "{},{},{},{}\n".format(win, current_cell, entry_cell, exit_cell)
                with open(output_file_path, 'w') as output_file:
                    output_file.write(new_lines)

    def window_entry_exit(self):
        file_list = sorted_files(self.f4_entry_exit)
        fm_file_name = get_file_path(self.f3_dm, file_list[0])
        fm_df = pd.read_csv(fm_file_name)
        last_window = len(fm_df)
        for interval in range(last_window):
            output_file_path = get_file_path(self.f5_win_entry_exit, win_txt(interval))
            if not path_exists(output_file_path):
                print(' > window_entry_exit:', interval)
                df = pd.DataFrame(columns=['win', 'id', 'cell', 'entry', 'exit'])
                for file_name in file_list:
                    file_index = file_list.index(file_name)
                    file_path = get_file_path(self.f4_entry_exit, file_name)
                    file_df = pd.read_csv(file_path)
                    file_df = file_df[file_df.win == interval]
                    file_df['id'] = file_index
                    df = df.append(file_df)
                df.to_csv(output_file_path, index=False)

    def contact_time(self):
        total_nodes = len(sorted_files(self.f4_entry_exit))
        for interval in range(self.start_window, self.end_window):
            output_file_path = get_file_path(self.f6_contact_time, win_txt(interval))
            if not path_exists(output_file_path):
                print(' > contact_time:', interval)
                contact_times = np.zeros(shape=(total_nodes, total_nodes))
                file_path = get_file_path(self.f5_win_entry_exit, win_txt(interval))
                file_df = pd.read_csv(file_path)
                for ix, ri in file_df.iterrows():
                    idi = ri.id
                    new_df = file_df[file_df.id != ri.id]
                    new_df = new_df[new_df.cell == ri.cell]
                    new_df = new_df[new_df.entry >= ri.entry]
                    new_df = new_df[new_df.entry < ri.exit]
                    for jx, rj in new_df.iterrows():
                        idj = rj.id
                        first_exit = min(ri.exit, rj.exit)
                        contact_interval = first_exit - rj.entry
                        contact_times[idi, idj] = contact_times[idi, idj] + contact_interval
                        contact_times[idj, idi] = contact_times[idj, idi] + contact_interval
                    del new_df
                del file_df
                contact_times_df = pd.DataFrame(contact_times)
                contact_times_df.to_csv(output_file_path, index=False)

    def image_metrics(self):
        for interval in range(self.start_window, self.end_window):
            mse_output_path = get_file_path(self.f7_metrics, metric_win_txt(ImageMetric.MSE, interval))
            ssim_output_path = get_file_path(self.f7_metrics, metric_win_txt(ImageMetric.SSIM, interval))
            ari_output_path = get_file_path(self.f7_metrics, metric_win_txt(ImageMetric.ARI, interval))
            if not path_exists(mse_output_path):
                print(' > image_metrics:', interval)
                samples = []
                for file_name in sorted_files(self.f3_dm):
                    file_path = get_file_path(self.f3_dm, file_name)
                    samples_from_window = self.sample_handler.get_samples(file_path, interval, interval + 1)
                    samples.append(samples_from_window[0])
                total_samples = len(samples)
                if total_samples > 0:
                    mse = np.zeros(shape=(total_samples, total_samples))
                    ssim = np.zeros(shape=(total_samples, total_samples))
                    ari = np.zeros(shape=(total_samples, total_samples))
                    for idx_i, image_i in enumerate(samples):
                        image_i = squeeze(image_i)
                        for idx_j, image_j in enumerate(samples):
                            image_j = squeeze(image_j)
                            if idx_i == idx_j:
                                continue
                            mse[idx_i, idx_j] = f_mse(image_i, image_j)
                            ssim[idx_i, idx_j] = f_ssim(image_i, image_j)
                            ari[idx_i, idx_j] = f_ari(image_i.reshape(self.width * self.height),
                                                      image_j.reshape(self.width * self.height))
                    mse_df = pd.DataFrame(mse)
                    mse_df.to_csv(mse_output_path, index=False)
                    ssim_df = pd.DataFrame(ssim)
                    ssim_df.to_csv(ssim_output_path, index=False)
                    ari_df = pd.DataFrame(ari)
                    ari_df.to_csv(ari_output_path, index=False)


def compute_baseline(dataset: Dataset):
    baseline = BaselineComputation(dataset)
    baseline.cell_entry_exit()
    baseline.window_entry_exit()
    baseline.contact_time()
    baseline.image_metrics()

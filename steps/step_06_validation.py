import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from functions.array_list_functions import above_diagonal
from functions.console_functions import print_k_step
from functions.data_frame_functions import df_base_same_diff, column_k, load_csv
from functions.files_paths_functions import path_exists, get_file_path, win_dir, win_txt, metric_win_txt
from functions.graphics_functions import normalize
from functions.running_functions import existing_contact_time
from graphics import ValidationHandler, tsne_scatter_plot
from helpers.files_paths_helper import ExportedFilesName, Path
from helpers.parameters_helper import SampleSelectionType, ImageMetric
from helpers.types_helper import Dataset, TypeLearning
from instances.validation_instances import K_RANGE


class Tsne:

    @staticmethod
    def tsne_data(encodings, path: str, start_window: int, end_window: int, validation_start: int):
        csv, png = ExportedFilesName.tsne_window(path, start_window, end_window, validation_start)
        if path_exists(csv):
            tsne_data = np.loadtxt(csv, delimiter=',')
        else:
            tsne = TSNE(n_components=2, init='pca', verbose=0)
            tsne_data = tsne.fit_transform(encodings)
            np.savetxt(csv, tsne_data, delimiter=',')
            tsne_scatter_plot(tsne_data, png, start_window, end_window)
        return tsne_data


class Clustering:

    def __init__(self, input_data: np.array):
        self.input_data = input_data

    def gmm(self, k: int):
        gmm = GaussianMixture(n_components=k, warm_start=True, max_iter=1000, random_state=1, n_init=10, verbose=0)
        gmm.fit(self.input_data)
        aic = gmm.aic(self.input_data)
        bic = gmm.bic(self.input_data)
        labels = gmm.predict(self.input_data)
        clusters = np.unique(labels)
        return clusters, labels, aic, bic


class Validation:

    def __init__(self, dataset: Dataset, type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        self.type_learning = type_learning
        self.sample_selection_type = sample_selection_type
        self.f8_contact_time = Path.f6_contact_time(dataset.name)
        self.f9_metrics = Path.f7_metrics(dataset.name)
        self.f10_results = Path.f8_results(dataset.name, type_learning, sample_selection_type)

    def window_validation(self, input_data: np.array, indices_list: np.array, window: int):
        path = get_file_path(self.f10_results, win_dir(window))
        df_contact_time, df_heatmap = df_base_same_diff(), df_base_same_diff()
        bigger_avg, k_bigger_avg = 0, None

        clustering = Clustering(input_data)
        validation_handler = ValidationHandler(path, window, self.type_learning, self.sample_selection_type)

        existing_contacts = existing_contact_time(self.f8_contact_time, window)

        contact_time_csv = get_file_path(path, ExportedFilesName.CONTACT_TIME_CSV.value)
        heatmap_csv = get_file_path(path, ExportedFilesName.HEATMAP_CSV.value)

        if path_exists(contact_time_csv) and path_exists(heatmap_csv):
            curves_contact_time = pd.read_csv(contact_time_csv, sep=',')
            curves_heatmap = pd.read_csv(heatmap_csv, sep=',')
            print_k_step(window, K_RANGE[-1], K_RANGE)
        else:
            for k in K_RANGE:
                print_k_step(window, k, K_RANGE)

                clusters, labels, aic, bic = clustering.gmm(k)
                validation_handler.append_k_score(k, aic, bic)

                contact_times_sources = self.contact_time(window, clusters, labels, indices_list)
                heatmaps_sources = self.heatmaps(window, clusters, labels, indices_list)

                df_contact_time[column_k(k)] = contact_times_sources
                df_heatmap[column_k(k)] = heatmaps_sources
                bigger_avg, k_bigger_avg = validation_handler.k_and_bigger_avg(contact_times_sources, bigger_avg,
                                                                               k_bigger_avg, k)
            curves_contact_time, curves_heatmap = validation_handler.plot_chosen_ks(df_contact_time, df_heatmap,
                                                                                    k_bigger_avg)
            curves_contact_time.to_csv(contact_time_csv, sep=',', index=False)
            curves_heatmap.to_csv(heatmap_csv, sep=',', index=False)

        validation_handler.plot_contact_time(curves_contact_time, existing_contacts)
        validation_handler.plot_heatmap(curves_heatmap)

        del clustering, validation_handler
        return curves_contact_time, curves_heatmap

    def contact_time(self, window: int, clusters: np.array, labels: np.array, indices_list: np.array):
        path = get_file_path(self.f8_contact_time, win_txt(window))
        each_contact_time = load_csv(path).to_numpy()
        non_noise_contact_time = each_contact_time[:, indices_list][indices_list, :]
        baseline = above_diagonal(non_noise_contact_time)
        same = self.same_cluster_contact_time(non_noise_contact_time, clusters, labels)
        diff = self.diff_cluster_contact_time(non_noise_contact_time, clusters, labels)
        contact_times_types = [baseline, same, diff]
        del non_noise_contact_time
        return contact_times_types

    @staticmethod
    def same_cluster_contact_time(non_noise_contact_time: np.array, clusters: np.array, labels: np.array):
        all_contact_times = np.array([])
        for cluster in clusters:
            cluster_ids = np.where(labels == cluster)[0]
            cluster_contact_times = non_noise_contact_time[:, cluster_ids][cluster_ids, :]
            cluster_contact_times = above_diagonal(cluster_contact_times)
            all_contact_times = np.append(all_contact_times, cluster_contact_times)
        return all_contact_times

    @staticmethod
    def diff_cluster_contact_time(non_noise_contact_time: np.array, clusters: np.array, labels: np.array):
        all_contact_times = np.array([])
        all_indices = np.arange(0, len(labels), 1)
        saw_nodes = np.array([])
        for cluster in clusters:
            cluster_ids = np.where(labels == cluster)[0]
            saw_nodes = np.append(saw_nodes, cluster_ids).astype(int)
            off_cluster_ids = np.delete(all_indices, saw_nodes)
            if len(off_cluster_ids) > 0:
                diff_cluster_contact_times = non_noise_contact_time[:, off_cluster_ids][cluster_ids, :]
                all_contact_times = np.append(all_contact_times, diff_cluster_contact_times)
        del all_indices, saw_nodes
        return all_contact_times

    def heatmaps(self, window: int, clusters: np.array, labels: np.array, indices_list: np.array):
        mse_path = get_file_path(self.f9_metrics, metric_win_txt(ImageMetric.MSE, window))
        ssim_path = get_file_path(self.f9_metrics, metric_win_txt(ImageMetric.SSIM, window))
        ari_path = get_file_path(self.f9_metrics, metric_win_txt(ImageMetric.ARI, window))
        mse = load_csv(mse_path).to_numpy()
        non_noise_mse = normalize(mse[:, indices_list][indices_list, :])
        ssim = load_csv(ssim_path).to_numpy()
        non_noise_ssim = normalize(ssim[:, indices_list][indices_list, :])
        ari = load_csv(ari_path).to_numpy()
        non_noise_ari = normalize(ari[:, indices_list][indices_list, :])
        baseline = pd.DataFrame({'mse': above_diagonal(non_noise_mse),
                                 'ssim': above_diagonal(non_noise_ssim),
                                 'ari': above_diagonal(non_noise_ari)})
        same = self.same_cluster_heatmaps(non_noise_mse, non_noise_ssim, non_noise_ari, clusters, labels)
        diff = self.diff_cluster_heatmaps(non_noise_mse, non_noise_ssim, non_noise_ari, clusters, labels)
        heatmaps_types = [baseline, same, diff]
        del non_noise_mse, non_noise_ssim, non_noise_ari
        return heatmaps_types

    @staticmethod
    def same_cluster_heatmaps(non_noise_mse, non_noise_ssim, non_noise_ari, clusters, labels):
        all_mse, all_ssim, all_ari = np.array([]), np.array([]), np.array([])
        for cluster in clusters:
            cluster_ids = np.where(labels == cluster)[0]
            same_mse = non_noise_mse[:, cluster_ids][cluster_ids, :]
            same_mse = above_diagonal(same_mse)
            all_mse = np.append(all_mse, same_mse)
            same_ssim = non_noise_ssim[:, cluster_ids][cluster_ids, :]
            same_ssim = above_diagonal(same_ssim)
            all_ssim = np.append(all_ssim, same_ssim)
            same_ari = non_noise_ari[:, cluster_ids][cluster_ids, :]
            same_ari = above_diagonal(same_ari)
            all_ari = np.append(all_ari, same_ari)
        df_metrics_same = pd.DataFrame({'mse': all_mse, 'ssim': all_ssim, 'ari': all_ari})
        return df_metrics_same

    @staticmethod
    def diff_cluster_heatmaps(non_noise_mse, non_noise_ssim, non_noise_ari, clusters, labels):
        all_mse, all_ssim, all_ari = np.array([]), np.array([]), np.array([])
        all_indices = np.arange(0, len(non_noise_mse), 1)
        saw_nodes = np.array([])
        for cluster in clusters:
            cluster_ids = np.where(labels == cluster)[0]
            saw_nodes = np.append(saw_nodes, cluster_ids).astype(int)
            off_cluster_ids = np.delete(all_indices, saw_nodes)
            if len(off_cluster_ids) > 0:
                diff_mse = non_noise_mse[:, off_cluster_ids][cluster_ids, :]
                all_mse = np.append(all_mse, diff_mse)
                diff_ssim = non_noise_ssim[:, off_cluster_ids][cluster_ids, :]
                all_ssim = np.append(all_ssim, diff_ssim)
                diff_ari = non_noise_ari[:, off_cluster_ids][cluster_ids, :]
                all_ari = np.append(all_ari, diff_ari)
        df_metrics_diff = pd.DataFrame({'mse': all_mse, 'ssim': all_ssim, 'ari': all_ari})
        del all_indices, saw_nodes
        return df_metrics_diff

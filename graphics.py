from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle as ms
import pandas as pd
import numpy as np

from functions.array_list_functions import arange
from functions.data_frame_functions import best_k_columns, sources, df_conf_int, new_sources
from functions.files_paths_functions import get_file_path, path_exists, win_dir
from functions.graphics_functions import set_x_ticks, image_metrics_list, contact_time_curves, colors, line_styles, \
    set_y_lim, heatmap_curves, fill_between, get_label
from helpers.files_paths_helper import ExportedFilesName, dir_exists_create, Path
from helpers.graphics_helper import GraphicLabels, GraphProps, GraphicLegends
from helpers.parameters_helper import SampleSelectionType, ImageMetric
from helpers.types_helper import TypeLearning, Dataset


class LossesHandler:

    def __init__(self, path, type_learning: TypeLearning):
        self.training_path = get_file_path(path, ExportedFilesName.TRAINING_LOSS.value)
        self.testing_path = get_file_path(path, ExportedFilesName.TESTING_LOSS.value)
        self.plot_path = get_file_path(path, ExportedFilesName.LOSSES_CURVE.value)
        self.type_learning = type_learning
        self.training_losses = np.array([])
        self.testing_losses = np.array([])

    def get_losses(self):
        return self.training_losses, self.testing_losses

    def append_fed(self, training_metrics, testing_metrics):
        training_loss = list(list(training_metrics.values())[2].values())[0]
        num_samples = list(list(training_metrics.values())[3].values())[0]
        self.training_losses = np.append(self.training_losses, training_loss)

        testing_loss = list(testing_metrics.values())[0]
        self.testing_losses = np.append(self.testing_losses, testing_loss)

        print('Samples: {} | TR loss: {} | TS loss: {}'.format(num_samples, training_loss, testing_loss))
        return self.training_losses, self.testing_losses

    def append_cen(self, training_losses: list, testing_losses: list):
        self.training_losses = np.array(training_losses)
        self.testing_losses = np.array(testing_losses)

    def save(self):
        self.training_losses.tofile(self.training_path, sep=',')
        self.testing_losses.tofile(self.testing_path, sep=',')
        self.plot()

    def load_fed(self, trained_rounds: int):
        if trained_rounds > 0:
            self.training_losses = np.fromfile(self.training_path, sep=',')
            self.testing_losses = np.fromfile(self.testing_path, sep=',')
            self.plot()
        return self.training_losses, self.testing_losses

    def load_cen(self):
        if path_exists(self.training_path) and path_exists(self.testing_path):
            self.training_losses = np.fromfile(self.training_path, sep=',')
            self.testing_losses = np.fromfile(self.testing_path, sep=',')
            self.plot()

    def plot(self):
        plt.figure(figsize=GraphProps.FIG_SIZE_LOSS.value)

        plt.ylabel(GraphicLabels.LOSS.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)

        plt.plot(range(1, len(self.training_losses) + 1), self.training_losses[:], '-',
                 linewidth=GraphProps.LINEWIDTH_2_5.value, label=GraphicLegends.TRAINING_LOSS)
        plt.plot(range(1, len(self.testing_losses) + 1), self.testing_losses[:], '--',
                 linewidth=GraphProps.LINEWIDTH_2_5.value, label=GraphicLegends.TESTING_LOSS)
        if self.type_learning == TypeLearning.CEN:
            plt.xlabel(GraphicLabels.EPOCHS.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.xticks(np.arange(0, len(self.training_losses) + 1, 25), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        else:
            plt.xlabel(GraphicLabels.ROUNDS.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.xticks(np.arange(0, len(self.training_losses) + 1, 5), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.legend(fontsize=GraphProps.FONT_SIZE_LEGEND.value)

        plt.ylim([0.01, 0.05])
        plt.yticks([0.01, 0.02, 0.03, 0.04, 0.05], fontsize=GraphProps.FONT_SIZE_LEGEND.value)

        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()


class ValidationHandler:

    def __init__(self, path: str, window: int, type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        self.type_learning = type_learning
        self.sample_selection_type = sample_selection_type
        self.path = dir_exists_create(path)
        self.window = window
        self.k_aics = []
        self.k_bics = []
        self.best_columns = None

    @staticmethod
    def k_and_bigger_avg(contact_times_sources: list, bigger_avg: float, k_bigger_avg: int, k: int):
        same_avg = contact_times_sources[1].mean()
        if same_avg > bigger_avg:
            return same_avg, k
        else:
            return bigger_avg, k_bigger_avg

    def append_k_score(self, k: int, aic: float, bic: float):
        self.k_aics.append([aic, k])
        self.k_bics.append([bic, k])

    def plot_chosen_ks(self, df_contact_time: pd.DataFrame, df_heatmap: pd.DataFrame, k_bigger_avg: int):
        self.k_bics.sort()
        self.k_aics.sort()
        self.best_columns = best_k_columns(df_contact_time, self.k_bics, self.k_aics, k_bigger_avg)

        curves_contact_time = contact_time_curves(df_contact_time[self.best_columns], self.best_columns)
        curves_heatmap = heatmap_curves(df_heatmap[self.best_columns], self.best_columns)

        return curves_contact_time, curves_heatmap

    def plot_contact_time(self, data_curves: pd.DataFrame, existing_contacts: int):
        plt.figure(figsize=GraphProps.FIG_SIZE_CONTACT_TIME.value)
        x_axis = arange(data_curves.columns)
        error_bar(x_axis, data_curves, existing_contacts)
        plt.ylabel(GraphicLabels.CONTACT_TIME_SEC.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        locs, labels = plt.yticks()
        plt.yticks(np.linspace(round(locs[0]), round(locs[-1]), num=5, dtype=int),
                   fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.xlabel(GraphicLabels.NUM_CLUSTERS.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.xticks(x_axis, set_x_ticks(data_curves.columns[1:]), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.legend(loc=GraphProps.LEGEND_LOC_BETTER.value, borderpad=GraphProps.BORDERPAD.value,
                   labelspacing=GraphProps.LABELSPACING.value, markerscale=GraphProps.LEGEND_MS.value,
                   fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.tight_layout()
        plt.savefig(get_file_path(self.path, ExportedFilesName.CONTACT_TIME.value))
        plt.close()

    def plot_heatmap(self, data_curves: pd.DataFrame):
        for metric in image_metrics_list():
            plt.figure(figsize=GraphProps.FIG_SIZE_HEATMAP.value)
            x_axis = arange(data_curves.columns[1:])
            df_metric = data_curves[data_curves.metric == metric]
            df_metric = df_metric.loc[:, df_metric.columns != 'metric']
            error_bar(x_axis, df_metric)
            plt.ylabel(GraphicLabels.metric(metric), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            locs, labels = plt.yticks()
            yticks = np.array([])
            for idx in np.linspace(locs[0], locs[-1], num=5):
                yticks = np.append(yticks, '{:.2f}'.format(idx))
            plt.yticks(yticks.astype(float), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.xticks(x_axis, set_x_ticks(data_curves.columns[2:]), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.xlabel(GraphicLabels.NUM_CLUSTERS.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.legend(loc=GraphProps.LEGEND_LOC_BETTER.value, borderpad=GraphProps.BORDERPAD.value,
                       labelspacing=GraphProps.LABELSPACING.value, markerscale=GraphProps.LEGEND_MS.value,
                       fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.tight_layout()
            plt.savefig(get_file_path(self.path, ExportedFilesName.METRICS.value.format(metric)))
            plt.close()

    @staticmethod
    def tsne_plot(tsne_data, clusters, labels):
        fig, ax = plt.subplots()
        markers = list(ms.markers.keys())
        for cluster in clusters:
            row_ix = np.where(labels == cluster)
            ax.scatter(tsne_data[row_ix, 0], tsne_data[row_ix, 1], marker=markers[cluster],
                       label='Community {}'.format(cluster + 1))
        fig.legend()
        plt.xlabel('t-SNE (X)')
        plt.ylabel('t-SNE (Y)')
        plt.show()


class IntervalEvolution:

    def __init__(self, path: str, type_learning: TypeLearning, sample_selection_type: SampleSelectionType):
        self.path = path
        self.type_learning = type_learning
        self.sample_selection_type = sample_selection_type
        self.dfs_curves_contact_time = []
        self.dfs_curves_heatmap = []
        self.k_list = []

    def append_curves(self, curves_contact_time: pd.DataFrame, curves_heatmap: pd.DataFrame):
        self.dfs_curves_contact_time.append(curves_contact_time)

        curves_heatmap = curves_heatmap[curves_heatmap.metric == image_metrics_list()[1]]
        curves_heatmap = curves_heatmap.loc[:, curves_heatmap.columns != 'metric']

        self.dfs_curves_heatmap.append(curves_heatmap)

    def best_k_interval(self, first_validate_interval: int, is_heatmap: bool = False):
        x_ticks_list = []
        interval = first_validate_interval

        baseline_ci = df_conf_int()
        same_ci = df_conf_int()
        diff_ci = df_conf_int()

        if is_heatmap:
            df_to_explore = self.dfs_curves_heatmap
        else:
            df_to_explore = self.dfs_curves_contact_time

        for data_frame in df_to_explore:
            baseline, same, diff, best_k = self.find_best_k(data_frame)
            baseline_ci['{}'.format(interval)] = np.nan_to_num(baseline)
            same_ci['{}'.format(interval)] = np.nan_to_num(same)
            diff_ci['{}'.format(interval)] = np.nan_to_num(diff)
            x_ticks_list.append('{}\n(k:{})'.format(interval, best_k))
            interval += 1
        self.plot_evolution(baseline_ci, same_ci, diff_ci, x_ticks_list, is_heatmap)

    def find_best_k(self, data_frame: pd.DataFrame):
        baseline = data_frame.iloc[0:3, -1].to_numpy()
        same = data_frame.iloc[3:6, -1].to_numpy()
        diff = data_frame.iloc[6:9, -1].to_numpy()
        best_k = int(float(data_frame.columns[-1][2:]))  # FIX: sometimes the column name becomes from k:3 to k:3.1
        self.k_list.append(best_k)
        return baseline, same, diff, best_k

    def most_common_k(self):
        self.k_list.sort(reverse=True)
        bigger = Counter(self.k_list).most_common(1)[0][0]
        return ['source', bigger]

    def plot_evolution(self, baseline_ci: pd.DataFrame, same_ci: pd.DataFrame, diff_ci: pd.DataFrame,
                       x_ticks_list: list, is_heatmap: bool = False):
        fig, ax = plt.subplots()
        fig.set_size_inches(GraphProps.FIG_SIZE_EVOLUTION.value)
        x_axis = arange(baseline_ci.columns)

        line_style = line_styles()
        min_baseline, max_baseline = fill_between(x_axis, baseline_ci, sources()[0], line_style.pop(), ax)
        min_same, max_same = fill_between(x_axis, same_ci, sources()[1], line_style.pop(), ax)
        min_diff, max_diff = fill_between(x_axis, diff_ci, sources()[2], line_style.pop(), ax)

        set_y_lim(min_baseline, min_same, min_diff, max_baseline, max_same, max_diff, plt.ylim)
        plt.xlabel(GraphicLabels.INTERVAL_INDICES.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.xticks(x_axis, x_ticks_list, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.legend(loc=GraphProps.LEGEND_LOC_BETTER.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        if is_heatmap:
            plt.ylabel(GraphicLabels.metric(image_metrics_list()[1]), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            locs, labels = plt.yticks()
            yticks = np.array([])
            for idx in np.linspace(locs[0], locs[-1], num=5):
                yticks = np.append(yticks, '{:.2f}'.format(idx))
            plt.yticks(yticks.astype(float), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.tight_layout()
            plt.savefig(get_file_path(self.path, ExportedFilesName.SSIM_EVOLUTION_PNG.value.format(
                self.type_learning.value, self.sample_selection_type.value
            )))
        else:
            plt.ylabel(GraphicLabels.CONTACT_TIME_SEC.value, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            locs, labels = plt.yticks()
            plt.yticks(np.linspace(round(locs[0]), round(locs[-1]), num=5, dtype=int),
                       fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            plt.tight_layout()
            plt.savefig(get_file_path(self.path, ExportedFilesName.CONTACT_TIME_EVOLUTION_PNG.value.format(
                self.type_learning.value, self.sample_selection_type.value
            )))
        plt.close()


class StrategiesMatch:

    def __init__(self, path):
        self.path = path
        self.best_path = get_file_path(path, ExportedFilesName.BEST_K_CSV.value)
        self.fixed_path = get_file_path(path, ExportedFilesName.FIXED_K_CSV.value)

    @staticmethod
    def match_selection_strategy_chosen_ks(dataset: Dataset, type_learning: TypeLearning, first_interval: int,
                                           last_interval: int):
        path_match = Path.f8_results_match(dataset.name, type_learning)
        path_acc = Path.f8_results(dataset.name, type_learning, SampleSelectionType.ACC)
        path_sli = Path.f8_results(dataset.name, type_learning, SampleSelectionType.SLI)
        choice = ['BIC', 'AIC', 'Best']
        for interval in range(first_interval, last_interval):
            acc_interval = get_file_path(path_acc, win_dir(interval))
            sli_interval = get_file_path(path_sli, win_dir(interval))
            acc_df = pd.read_csv(get_file_path(acc_interval, ExportedFilesName.CONTACT_TIME_CSV.value))
            sli_df = pd.read_csv(get_file_path(sli_interval, ExportedFilesName.CONTACT_TIME_CSV.value))
            k_list = []

            mean_baseline = sli_df.iloc[0][1]

            acc_columns = acc_df.columns[1:]
            sli_columns = sli_df.columns[1:]
            for index, item in enumerate(choice):
                acc_column = int(float(acc_columns[index][2:]))  # FIX: sometimes the column name becomes from k:3 to
                # k:3.1
                sli_column = int(float(sli_columns[index][2:]))  # FIX: sometimes the column name becomes from k:3 to
                # k:3.1
                k_list.append('ACC k:{} \nSLI k: {}'.format(acc_column, sli_column))
            labels = ['{}\n{}'.format(choice[i], item) for i, item in enumerate(k_list)]
            x_labels = np.arange(len(labels))
            width = 0.1

            sames_acc, diffs_acc = [], []
            sames_sli, diffs_sli = [], []
            for index, item in enumerate(choice):
                acc_same = float(acc_df.iloc[3][index + 1])
                acc_diff = float(acc_df.iloc[6][index + 1])

                sli_same = float(sli_df.iloc[3][index + 1])
                sli_diff = float(sli_df.iloc[6][index + 1])

                sames_acc.append(acc_same)
                sames_sli.append(sli_same)
                diffs_acc.append(acc_diff)
                diffs_sli.append(sli_diff)

            fig, ax = plt.subplots()
            fig.set_size_inches(GraphProps.FIG_SIZE_STRATEGY_MATCH.value)
            color_list = colors()[:-1]
            plt.axhline(mean_baseline, color=color_list.pop(), linestyle='dashed', label=new_sources()[0])
            ax.bar(x_labels - (3 * width / 2), sames_acc, width, color=color_list.pop(),
                   label='{} ACC'.format(new_sources()[1]))
            ax.bar(x_labels - (1 * width / 2), diffs_acc, width, color=color_list.pop(),
                   label='{} ACC'.format(new_sources()[2]))
            ax.bar(x_labels + (1 * width / 2), sames_sli, width, color=color_list.pop(),
                   label='{} SLI'.format(new_sources()[1]))
            ax.bar(x_labels + (3 * width / 2), diffs_sli, width, color=color_list.pop(),
                   label='{} SLI'.format(new_sources()[2]))

            ax.set_ylabel(GraphicLabels.AVG_CONTACT_TIME_SEC, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.set_xlabel(GraphicLabels.K_VALUES, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.set_xticks(x_labels)
            ax.set_xticklabels(labels, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.tick_params(axis='both', which='both', labelsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.legend(fontsize=GraphProps.FONT_SIZE_LEGEND.value, loc=GraphProps.LEGEND_LOC_BETTER.value)

            min_baseline, min_same, min_diff = mean_baseline, min(sames_sli + sames_acc), min(diffs_sli + diffs_acc)
            max_baseline, max_same, max_diff = mean_baseline, max(sames_sli + sames_acc), max(diffs_sli + diffs_acc)

            set_y_lim(min_baseline, min_same, min_diff, max_baseline, max_same, max_diff, ax.set_ylim)
            plt.tight_layout()
            plt.savefig(get_file_path(path_match, ExportedFilesName.interval_match(interval)))
            plt.close()

        del acc_df, sli_df

    @staticmethod
    def ssim_match_selection_strategy_chosen_ks(dataset: Dataset, type_learning: TypeLearning, first_interval: int,
                                                last_interval: int):
        path_match = Path.f8_results_match(dataset.name, type_learning)
        path_acc = Path.f8_results(dataset.name, type_learning, SampleSelectionType.ACC)
        path_sli = Path.f8_results(dataset.name, type_learning, SampleSelectionType.SLI)
        choice = ['BIC', 'AIC', 'Best']
        for interval in range(first_interval, last_interval):
            acc_interval = get_file_path(path_acc, win_dir(interval))
            sli_interval = get_file_path(path_sli, win_dir(interval))
            acc_df = pd.read_csv(get_file_path(acc_interval, ExportedFilesName.HEATMAP_CSV.value))
            sli_df = pd.read_csv(get_file_path(sli_interval, ExportedFilesName.HEATMAP_CSV.value))
            k_list = []

            mean_baseline = sli_df.iloc[9][2]

            acc_columns = acc_df.columns[2:]
            sli_columns = sli_df.columns[2:]
            for index, item in enumerate(choice):
                acc_column = int(float(acc_columns[index][2:]))  # FIX: sometimes the column name becomes from k:3 to
                # k:3.1
                sli_column = int(float(sli_columns[index][2:]))  # FIX: sometimes the column name becomes from k:3 to
                # k:3.1
                k_list.append('ACC k:{} \nSLI k: {}'.format(acc_column, sli_column))
            labels = ['{}\n{}'.format(choice[i], item) for i, item in enumerate(k_list)]
            x_labels = np.arange(len(labels))
            width = 0.1

            sames_acc, diffs_acc = [], []
            sames_sli, diffs_sli = [], []
            for index, item in enumerate(choice):
                acc_same = float(acc_df.iloc[12][index + 2])
                acc_diff = float(acc_df.iloc[15][index + 2])

                sli_same = float(sli_df.iloc[12][index + 2])
                sli_diff = float(sli_df.iloc[15][index + 2])

                sames_acc.append(acc_same)
                sames_sli.append(sli_same)
                diffs_acc.append(acc_diff)
                diffs_sli.append(sli_diff)

            fig, ax = plt.subplots()
            fig.set_size_inches(GraphProps.FIG_SIZE_STRATEGY_MATCH.value)
            color_list = colors()[:-1]
            plt.axhline(mean_baseline, color=color_list.pop(), linestyle='dashed', label=new_sources()[0])
            ax.bar(x_labels - (3 * width / 2), sames_acc, width, color=color_list.pop(),
                   label='{} ACC'.format(new_sources()[1]))
            ax.bar(x_labels - (1 * width / 2), diffs_acc, width, color=color_list.pop(),
                   label='{} ACC'.format(new_sources()[2]))
            ax.bar(x_labels + (1 * width / 2), sames_sli, width, color=color_list.pop(),
                   label='{} SLI'.format(new_sources()[1]))
            ax.bar(x_labels + (3 * width / 2), diffs_sli, width, color=color_list.pop(),
                   label='{} SLI'.format(new_sources()[2]))

            ax.set_ylabel(GraphicLabels.metric(ImageMetric.SSIM.value), fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.set_xlabel(GraphicLabels.K_VALUES, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.set_xticks(x_labels)
            ax.set_xticklabels(labels, fontsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.tick_params(axis='both', which='both', labelsize=GraphProps.FONT_SIZE_LEGEND.value)
            ax.legend(fontsize=GraphProps.FONT_SIZE_LEGEND.value, loc=GraphProps.LEGEND_LOC_BETTER.value)

            min_baseline, min_same, min_diff = mean_baseline, min(sames_sli + sames_acc), min(diffs_sli + diffs_acc)
            max_baseline, max_same, max_diff = mean_baseline, max(sames_sli + sames_acc), max(diffs_sli + diffs_acc)

            set_y_lim(min_baseline, min_same, min_diff, max_baseline, max_same, max_diff, ax.set_ylim)
            plt.tight_layout()
            plt.savefig(get_file_path(path_match, ExportedFilesName.ssim_interval_match(interval)))
            plt.close()

        del acc_df, sli_df


def error_bar(x_axis: np.array, data_curves: pd.DataFrame, existing_contacts: int = None):
    fmts = ['+--', 'x:', 'o-']
    for source in sources():
        aux = data_curves[data_curves.source == source].to_numpy()[:, 1:]
        mean = aux[0].astype(float)
        p025 = aux[1].astype(float)
        p975 = aux[2].astype(float)
        plt.errorbar(x_axis, mean, yerr=[mean - p025, p975 - mean], capsize=GraphProps.CAP_SIZE.value, fmt=fmts.pop(),
                     ms=GraphProps.MS.value, marker=GraphProps.MARKER.value, capthick=GraphProps.CAP_THICK.value,
                     elinewidth=GraphProps.LINEWIDTH_2.value, label=get_label(source))
    extra_plot = plt.plot([], [], color='white')
    if existing_contacts is not None:
        info = plt.legend(extra_plot, ['Number of contacts: {}'.format(existing_contacts)],
                          loc=GraphProps.EXTRA_LOC.value, handletextpad=GraphProps.HANDLETEXTPAD.value,
                          fontsize=GraphProps.FONT_SIZE_LEGEND.value)
        plt.gca().add_artist(info)


def tsne_scatter_plot(tsne_data: np.array, path: str, start_window: int, end_window: int):
    plt.figure(figsize=GraphProps.FIG_SIZE_REGULAR.value)
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
    plt.xlabel('t-SNE (X)')
    plt.ylabel('t-SNE (Y)')
    plt.suptitle('t-SNE data of interval {} from \n'
                 'training window [{}, {})'.format(end_window, start_window, end_window))
    plt.savefig(path)
    plt.close()


def pdf_plot(data: np.array):
    sns.kdeplot(data, color='red')
    plt.show()

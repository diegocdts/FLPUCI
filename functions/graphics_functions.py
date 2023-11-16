import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import axes
from scipy import stats, interpolate

from functions.data_frame_functions import df_data_curve, sources, new_sources
from helpers.parameters_helper import ImageMetric


def image_metrics_list():
    return [metric.value for metric in ImageMetric]


def confidence_interval(data):
    mean, sigma = np.mean(data), np.std(data)
    conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma / np.sqrt(len(data)))
    return conf_int


def set_x_ticks(x_ticks: np.array):
    technique = ['BIC', 'AIC', 'Best']
    new_x_ticks = np.array([])
    for index, item in enumerate(x_ticks):
        new_x_ticks = np.append(new_x_ticks, '{} ({})'.format(technique[index], item))
    return new_x_ticks


def set_y_lim(min_baseline: float, min_same: float, min_diff: float, max_baseline: float, max_same: float,
              max_diff: float, to_set: any):
    minimum = min(min_baseline, min_same, min_diff)
    if minimum > 0.0:
        minimum = minimum - (0.1 * minimum)
    else:
        minimum = 0.0
    maximum = max(max_baseline, max_same, max_diff)
    maximum = maximum + (0.1 * maximum)
    to_set([minimum, maximum])


def get_source_value(source_value: str):
    content = source_value.replace('[', '').replace(']', '')
    content_split = content.split(',')
    source, value = content_split[0].replace('\'', ''), round(float(content_split[1]), 2)
    return source, value


def contact_time_curves(df_contact_time: pd.DataFrame, columns: list):
    data_curves = df_data_curve(columns)
    data_index = 0
    for index, row in df_contact_time.iterrows():
        mean, p025, p975 = np.array([row.source]), np.array([row.source]), np.array([row.source])
        for column in df_contact_time.columns[1:]:
            data_k = row[column]
            if type(data_k) is pd.Series:
                data_k = data_k[0]
            mean = np.append(mean, data_k.mean())
            p025 = np.append(p025, confidence_interval(data_k)[0])
            p975 = np.append(p975, confidence_interval(data_k)[1])
        data_curves.loc[data_index] = mean
        data_curves.loc[data_index + 1] = p025
        data_curves.loc[data_index + 2] = p975
        data_index += 3
    return data_curves


def heatmap_curves(df_heatmap: pd.DataFrame, columns: list):
    data_curves = df_data_curve(columns, is_heatmap=True)
    data_index = 0
    for metric in image_metrics_list():
        for index, row in df_heatmap.iterrows():
            source = row.source
            mean, p025, p975 = np.array([metric, source]), np.array([metric, source]), np.array([metric, source])
            for column in df_heatmap.columns[1:]:
                data_k = row[column]
                if type(data_k) is pd.Series:
                    data_k = data_k[0]
                data_k = data_k[metric]
                mean = np.append(mean, data_k.mean())
                p025 = np.append(p025, confidence_interval(data_k)[0])
                p975 = np.append(p975, confidence_interval(data_k)[1])
            data_curves.loc[data_index] = mean
            data_curves.loc[data_index + 1] = p025
            data_curves.loc[data_index + 2] = p975
            data_index += 3
    return data_curves


def normalize(data: np.array):
    flatten = data.flatten()
    minimum = min(flatten)
    maximum = max(flatten)
    return (data - minimum) / (maximum - minimum)


def bspline_plot(x: np.array, y: np.array):
    x_new = np.linspace(int(x[0]), int(x[-1]), 100)
    bspline = interpolate.make_interp_spline(x, y, 2)
    y_new = bspline(x_new)
    return x_new, y_new


def fill_between(x_axis: np.array, df_ci: pd.DataFrame, source: str, line_style: str, ax: axes.Axes):
    x, mean = bspline_plot(x_axis, df_ci.iloc[0][1:].to_numpy())
    x, p025 = bspline_plot(x_axis, df_ci.iloc[1][1:].to_numpy())
    x, p975 = bspline_plot(x_axis, df_ci.iloc[2][1:].to_numpy())

    ax.fill_between(x, p975, p025, alpha=.5, linewidth=0)
    ax.plot(x, mean, linewidth=4, label=get_label(source), linestyle=line_style)

    return min(p025), max(p975)


def colors():
    return list(mcolors.TABLEAU_COLORS.values())[::-1]


def line_styles():
    return ['dashed', 'dotted', 'dashed', 'dotted', 'solid']


def get_label(source: str):
    if source == sources()[0]:
        return new_sources()[0]
    if source == sources()[1]:
        return new_sources()[1]
    if source == sources()[2]:
        return new_sources()[2]
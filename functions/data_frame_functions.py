import pandas as pd


def load_csv(path):
    return pd.read_csv(path)


def sources():
    return ['baseline', 'same cluster', 'diff clusters']


def new_sources():
    return ['All pairs', 'Intra-community', 'Inter-community']


def df_base_same_diff():
    return pd.DataFrame(columns=['source'], data=sources())


def df_data_curve(columns: list, is_heatmap: bool = False):
    if is_heatmap:
        columns = ['metric'] + columns
    return pd.DataFrame(columns=columns)


def df_conf_int():
    return pd.DataFrame(columns=['ci'], data=['mean', 'p025', 'p975'])


def column_k(k: int):
    return 'k:{}'.format(k)


def best_k_columns(df_contact_time: pd.DataFrame, k_bics: list, k_aics: list, k_bigger_avg: int):
    k_bic = get_k_score_column(df_contact_time, k_bics)
    k_aic = get_k_score_column(df_contact_time, k_aics)
    return ['source', k_bic, k_aic, column_k(k_bigger_avg)]


def get_k_score_column(df_contact_time: pd.DataFrame, k_scores: list):
    baseline_mean = df_contact_time.iloc[0][1].mean()
    best_score_column = column_k(k_scores[0][1])
    for k_score in k_scores:
        column = column_k(k_score[1])
        same_mean = df_contact_time.iloc[1][column].mean()
        if same_mean >= baseline_mean + (baseline_mean * 0.05):
            best_score_column = column
            break
    return best_score_column

from datetime import datetime
import os


def time():
    now = datetime.now()
    return '{}:{}:{}'.format(now.hour, now.minute, now.second)


def progress_bar(progress: int, total: int):
    #terminal_size = os.get_terminal_size().columns
    terminal_size = 100
    percent_bar = (terminal_size / 3) * (progress / float(total))
    percent_value = 100 * (progress / float(total))
    bar = '>' * int(percent_bar) + '-' * (int((terminal_size / 3)) - int(percent_bar))
    return f"|{bar}| {percent_value:.2f}%"


def print_k_step(window: int, k: int, k_range: range):
    print('\r> Validation at window: {} {}'.format(window, progress_bar(progress=k - 1, total=len(k_range))), end='\r')
    if k == k_range[-1]:
        print('\n')
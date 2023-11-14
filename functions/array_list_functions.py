import numpy as np


def squeeze(array: np.array):
    return np.squeeze(array)


def above_diagonal(matrix: np.array):
    return matrix[np.triu_indices(len(matrix), k=1)]


def arange(array: np.array):
    return np.arange(0, len(array) - 1)


def discrete_pixels(pixels: np.array):
    pixels = pixels * 255
    pixels = pixels.astype("int64")
    return pixels


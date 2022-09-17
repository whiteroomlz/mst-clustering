import math
import numpy as np

from numpy import ndarray as ndarr
from numba import njit


@njit(cache=True)
def fuzzy_covariance_matrix(data: ndarr, weighting_exp: float, partition: ndarr, cluster_center: ndarr) -> ndarr:
    covariance_matrix = np.zeros((data.shape[1], data.shape[1]))

    diff_matrix = data - cluster_center
    partition_sum = 0

    for index in np.arange(diff_matrix.shape[0]):
        diff_vector = np.expand_dims(diff_matrix[index, :], 0).T
        partition_ratio = partition[index] ** weighting_exp
        covariance_matrix += partition_ratio * diff_vector @ diff_vector.T
        partition_sum += partition_ratio
    covariance_matrix /= partition_sum

    return covariance_matrix


@njit(cache=True)
def hyper_volume(data: ndarr, weighting_exponent: float, cluster_ids: ndarr, cluster_center: ndarr) -> float:
    partition = np.zeros(data.shape[0])
    partition[cluster_ids] = 1

    det = np.linalg.det(fuzzy_covariance_matrix(data, weighting_exponent, partition, cluster_center))

    if det <= 0:
        return math.inf
    else:
        return np.sqrt(det)


@njit(cache=True)
def fuzzy_hyper_volume(data: ndarr, weighting_exponent: float, partition: ndarr, cluster_center: ndarr) -> float:
    det = np.linalg.det(fuzzy_covariance_matrix(data, weighting_exponent, partition, cluster_center))

    if det <= 0:
        return math.inf
    else:
        return np.sqrt(det)


def cluster_ln_distances(data: ndarr, weighting_exponent: float, partition: ndarr, cluster: int) -> ndarr:
    partition_ratio = partition[cluster] ** weighting_exponent
    cluster_center = partition_ratio @ data / np.sum(partition_ratio)

    covariance_matrix = fuzzy_covariance_matrix(data, weighting_exponent, partition[cluster], cluster_center)

    priory_probability = np.sum(partition_ratio) / data.shape[0]

    distances = np.ones(data.shape[0])
    for point_index in np.arange(data.shape[0]):
        diff_vector = (data[point_index] - cluster_center)[np.newaxis, :].T
        double_exp_power = diff_vector.T @ np.linalg.inv(covariance_matrix) @ diff_vector
        square_fhv = np.linalg.det(covariance_matrix)
        ln_distance = 0.5 * (data.shape[0] * np.log(2 * np.pi) + np.log(square_fhv) + double_exp_power) - np.log(
            priory_probability)
        distances[point_index] = ln_distance

    return distances


@njit(cache=True)
def zero_axis_sum(data):
    return np.sum(data, axis=0)

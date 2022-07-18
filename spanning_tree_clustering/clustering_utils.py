import numpy as np
from decimal import Decimal


def fuzzy_covariance_matrix(data: np.ndarray, weighting_exponent: float, partition: np.ndarray,
                            cluster_center: np.ndarray) -> np.ndarray:
    covariance_matrix = np.zeros((data.shape[1], data.shape[1]))

    diff_matrix = data - cluster_center
    partition_sum = 0

    for index in np.arange(diff_matrix.shape[0]):
        diff_vector = np.expand_dims(diff_matrix[index, :], 0).T
        partition_ratio = partition[index] ** weighting_exponent
        covariance_matrix += partition_ratio * diff_vector @ diff_vector.T
        partition_sum += partition_ratio
    covariance_matrix /= partition_sum

    return covariance_matrix


def fuzzy_hyper_volume(data: np.ndarray, weighting_exponent: float, cluster_ids: np.ndarray,
                       cluster_center: np.ndarray) -> float:
    partition = np.zeros(data.shape[0])
    partition[cluster_ids] = 1

    det = np.linalg.det(fuzzy_covariance_matrix(data, weighting_exponent, partition, cluster_center))

    if det <= 0:
        return -1
    else:
        return np.sqrt(det)


def cluster_distances(data: np.ndarray, weighting_exponent: float, partition: np.ndarray, cluster: int) -> np.ndarray:
    count_of_points = data.shape[0]
    partition_ratio = partition[cluster] ** weighting_exponent
    cluster_center = np.sum((data.T * partition_ratio).T, axis=0) / np.sum(partition_ratio)

    covariance_matrix = fuzzy_covariance_matrix(data, weighting_exponent, partition[cluster], cluster_center)
    det = np.linalg.det(covariance_matrix)

    distances = np.ones(count_of_points, dtype=Decimal)

    if det > 0:
        priory_probability = Decimal(np.sum(partition_ratio) / count_of_points)
        fhv = Decimal(np.sqrt(det))

        for point_index in np.arange(count_of_points):
            diff_vector = (data[point_index] - cluster_center)[np.newaxis, :].T
            distance = (fhv * ((Decimal(2 * np.pi)) ** Decimal(0.5 * count_of_points))) / priory_probability
            distance *= Decimal(
                float(0.5 * (diff_vector.T @ np.linalg.inv(covariance_matrix) @ diff_vector))
            ).exp()
            distances[point_index] = distance
    else:
        raise ValueError("Poor initial approximation.")

    return distances

import ctypes
import math

import numpy as np

from mst_clustering.math_utils import fuzzy_hyper_volume, cluster_distances
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from mst_clustering.cpp_adapters import SpanningForest, Edge
from multiprocessing.sharedctypes import RawArray, RawValue
from abc import ABC, abstractmethod
from decimal import Decimal


class ClusteringModel(ABC):
    @abstractmethod
    def __call__(self, data: np.ndarray, forest: SpanningForest, workers: int = 1, partition: np.ndarray = None) \
            -> np.ndarray:
        pass

    @staticmethod
    def get_cluster_info(data: np.ndarray, forest: SpanningForest, cluster_idx: int) -> (np.ndarray, list, np.ndarray):
        root = forest.get_roots()[cluster_idx]
        cluster_edges = forest.get_edges(root)

        if len(cluster_edges) == 0:
            cluster_ids = np.array([root])
            cluster_center = data[cluster_ids.squeeze()]
        else:
            cluster_ids = np.unique(list(map(lambda edge: [edge.first_node, edge.second_node], cluster_edges)))
            cluster_center = np.mean(data[cluster_ids], axis=0)

        return cluster_ids, cluster_edges, cluster_center


class ZahnModel(ClusteringModel):
    cutting_cond: float
    weighting_exp: float
    fhv_condition: float
    num_of_clusters: int
    use_first_criterion: bool
    use_second_criterion: bool
    use_third_criterion: bool
    use_additional_criterion: bool

    def __init__(self, cutting_condition, weighting_exponent, fhv_condition, num_of_clusters: int = -1,
                 use_first_criterion: bool = True, use_second_criterion: bool = True,
                 use_third_criterion: bool = True, use_additional_criterion: bool = True):
        self.cutting_cond = cutting_condition
        self.weighting_exp = weighting_exponent
        self.fhv_condition = fhv_condition
        self.num_of_clusters = num_of_clusters
        self.use_first_criterion = use_first_criterion
        self.use_second_criterion = use_second_criterion
        self.use_third_criterion = use_third_criterion
        self.use_additional_criterion = use_additional_criterion

    def __call__(self, data: np.ndarray, forest: SpanningForest, workers: int = 1, partition: np.ndarray = None) -> \
            np.ndarray:
        pool_args = (dict({
            "shared_data": RawArray(ctypes.c_double, data.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, data.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exp)
        }),)

        with ProcessPoolExecutor(max_workers=workers, initializer=pool_init, initargs=pool_args) as pool:
            while self._check_num_of_clusters(forest):
                info = list(map(lambda c: ZahnModel.get_cluster_info(data, forest, c), range(forest.size)))
                futures = list(pool.submit(parallel_fuzzy_hyper_volume, ids, center) for ids, _, center in info)

                wait(futures, return_when=ALL_COMPLETED)

                fuzzy_volumes = np.fromiter(map(lambda future: future.result(), futures), dtype=np.float64)
                fuzzy_volumes_without_noise = np.where(fuzzy_volumes == math.inf, -1, fuzzy_volumes)

                bad_cluster_edges = forest.get_edges(forest.get_roots()[np.argmax(fuzzy_volumes_without_noise)])

                weights = np.fromiter(map(lambda edge: edge.weight, bad_cluster_edges), dtype=np.float64)
                max_weight_idx = int(np.argmax(weights))
                max_weight = weights[max_weight_idx]

                if self.use_first_criterion and self._check_first_criterion(data, forest, max_weight):
                    worst_edge = bad_cluster_edges[max_weight_idx]
                elif self.use_second_criterion and self._check_second_criterion(weights, max_weight_idx):
                    worst_edge = bad_cluster_edges[max_weight_idx]
                elif self.use_third_criterion and (output := self._check_third_criterion(data, bad_cluster_edges)):
                    worst_edge = output
                else:
                    break

                forest.remove_edge(worst_edge.first_node, worst_edge.second_node)

        partition = np.zeros((forest.size, data.shape[0]))
        for cluster in range(forest.size):
            cluster_ids, _, _ = ZahnModel.get_cluster_info(data, forest, cluster)
            partition[cluster, cluster_ids] = 1

        return partition

    def _check_num_of_clusters(self, forest):
        return self.num_of_clusters == -1 or forest.size < self.num_of_clusters

    def _check_first_criterion(self, data: np.ndarray, forest: SpanningForest, edge_weight: float) -> bool:
        all_edges = forest.get_edges(forest.get_roots()[0])
        criterion = self.cutting_cond * sum(map(lambda edge: edge.weight, all_edges)) / (data.shape[0] - 1)
        return edge_weight >= criterion

    def _check_second_criterion(self, edges_weights: np.ndarray, edge_index: int) -> bool:
        weight = edges_weights[edge_index]
        edges_weights = np.delete(edges_weights, edge_index)
        return weight / np.mean(edges_weights) >= self.cutting_cond

    def _check_third_criterion(self, data: np.ndarray, cluster_edges: list) -> Edge or None:
        bad_edge_index = None

        temp_forest = SpanningForest(data.shape[0])
        for cluster_edge in cluster_edges:
            temp_forest.add_edge(cluster_edge.first_node, cluster_edge.second_node, cluster_edge.weight)

        min_total_fhv = math.inf
        for edge_index, cluster_edge in enumerate(cluster_edges):
            temp_forest.remove_edge(cluster_edge.first_node, cluster_edge.second_node)

            roots = temp_forest.get_roots()

            left_root = temp_forest.find_root(cluster_edge.first_node)
            left_cluster_ids, _, cluster_center = ZahnModel.get_cluster_info(data, temp_forest, roots.index(left_root))
            left_fhv = fuzzy_hyper_volume(data, self.weighting_exp, left_cluster_ids, cluster_center)

            right_root = temp_forest.find_root(cluster_edge.second_node)
            right_cluster_ids, _, cluster_center = self.get_cluster_info(data, temp_forest, roots.index(right_root))
            right_fhv = fuzzy_hyper_volume(data, self.weighting_exp, right_cluster_ids, cluster_center)

            if not (left_fhv is math.inf or right_fhv is math.inf):
                if (total_fhv := left_fhv + right_fhv) <= min_total_fhv:
                    bad_edge_index = edge_index
                    min_total_fhv = total_fhv

            temp_forest.add_edge(cluster_edge.first_node, cluster_edge.second_node, cluster_edge.weight)

        return cluster_edges[bad_edge_index] if min_total_fhv > self.fhv_condition else None


class GathGevaModel(ClusteringModel):
    termination_tolerance: float
    weighting_exp: float

    def __init__(self, termination_tolerance: float, weighting_exponent: float):
        self.termination_tolerance = termination_tolerance
        self.weighting_exp = weighting_exponent

    def __call__(self, data: np.ndarray, forest: SpanningForest, workers: int = 1, partition: np.ndarray = None) -> \
            np.ndarray:
        assert partition is not None, "This clustering method requires a non None partition matrix."

        initial_clusters_count = partition.shape[0]
        previous_partition = np.zeros_like(partition)

        while np.linalg.norm(partition - previous_partition) > self.termination_tolerance:
            previous_partition = partition.copy()
            power = Decimal(2 / (self.weighting_exp - 1))

            distance_matrix = self._get_distance_matrix(data, partition, workers)

            for cluster in np.arange(partition.shape[0]):
                for point_index in np.arange(partition.shape[1]):
                    distance = distance_matrix[cluster, point_index]
                    partition = sum(map(lambda other_cluster:
                                        (distance / distance_matrix[other_cluster, point_index]) ** power,
                                        np.arange(initial_clusters_count)))
                    partition **= -1
                    partition[cluster, point_index] = partition

        return partition

    def _get_distance_matrix(self, data: np.ndarray, partition: np.ndarray, workers: int) -> np.ndarray:
        pool_args = (dict({
            "shared_data": RawArray(ctypes.c_double, data.flatten()),
            "shared_partition": RawArray(ctypes.c_double, partition.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, data.shape[0]),
            "shared_clusters_count": RawValue(ctypes.c_int32, partition.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exp)
        }),)

        with ProcessPoolExecutor(max_workers=workers, initializer=pool_init, initargs=pool_args) as pool:
            distance_matrix = np.vstack(list(pool.map(parallel_compute_distances, range(partition.shape[0]))))

        return distance_matrix


# Parallel computation block.

shared_memory: dict


def pool_init(pool_args: dict):
    global shared_memory
    shared_memory = pool_args.copy()


def parallel_fuzzy_hyper_volume(cluster_ids: np.ndarray, cluster_center: np.ndarray) -> float:
    global shared_memory
    shared_data = shared_memory["shared_data"]
    shared_rows_count = shared_memory["shared_rows_count"]
    shared_weighting_exponent = shared_memory["shared_weighting_exponent"]

    data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
    weighting_exponent = shared_weighting_exponent.value
    return fuzzy_hyper_volume(data, weighting_exponent, cluster_ids, cluster_center)


def parallel_compute_distances(cluster: int) -> np.ndarray:
    global shared_memory
    shared_data = shared_memory["shared_data"]
    shared_partition = shared_memory["shared_partition"]
    shared_rows_count = shared_memory["shared_rows_count"]
    shared_clusters_count = shared_memory["shared_clusters_count"]
    shared_weighting_exponent = shared_memory["shared_weighting_exponent"]

    data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
    weighting_exponent = shared_weighting_exponent.value
    partition = np.frombuffer(shared_partition).reshape((shared_clusters_count.value, -1))
    return cluster_distances(data, weighting_exponent, partition, cluster)

import ctypes
import numpy as np

from spanning_tree_clustering.clustering_utils import fuzzy_hyper_volume, cluster_distances
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from spanning_tree_clustering.cpp_utils import SpanningForest, Edge
from multiprocessing.sharedctypes import RawArray, RawValue
from sklearn.preprocessing import normalize
from collections import defaultdict
from numba import njit, prange
from decimal import Decimal

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class SpanningTreeClustering(object):
    cutting_condition: float
    weighting_exponent: float
    termination_tolerance: float
    num_of_workers: int
    clustering_algorithm: str

    __data: np.ndarray
    __forest: SpanningForest
    __partition: np.ndarray
    __noise_occurrences: set

    def __init__(self, cutting_cond: float, weighting_exp: float, termination_tolerance: float,
                 num_of_workers: int = 1, clustering_algorithm: str = "hybrid"):
        self.cutting_condition = cutting_cond
        self.weighting_exponent = weighting_exp
        self.termination_tolerance = termination_tolerance
        self.num_of_workers = num_of_workers
        self.clustering_algorithm = clustering_algorithm

    def fit(self, data: np.ndarray, num_of_clusters):
        self.num_of_clusters = num_of_clusters

        self.__data = normalize(data)
        self.__forest = SpanningTreeClustering.build_mst(self.__data)

        if not self.__forest.is_spanning_tree():
            raise ValueError("There should be only a MST in the spanning forest.")

        self.__noise_occurrences = set()

        if self.clustering_algorithm == "simple" or self.clustering_algorithm == "hybrid":
            self.__partition = self.__simple_clustering()

        if self.clustering_algorithm == "gath-geva" or self.clustering_algorithm == "hybrid":
            # TODO: fix method.
            self.__partition = self.__gath_geva_algorithm()

        return self

    def get_labels(self) -> np.ndarray:
        labels = np.argmax(self.__partition, axis=0)
        labels_with_noise = np.where(labels == self.__noise_occurrences, -1, labels)
        return labels_with_noise

    def get_partition(self) -> np.ndarray:
        return self.__partition

    def get_clusters_count(self) -> np.ndarray:
        return self.__partition.shape[0]

    @staticmethod
    def build_mst(data: np.ndarray) -> SpanningForest:
        forest = SpanningForest(data.shape[0])

        nodes = np.arange(data.shape[0])
        is_node_used = np.zeros((data.shape[0],), dtype=np.bool)
        min_weight = np.ones((data.shape[0],), dtype=np.float64) * np.inf
        min_weight[0] = 0
        best_neighbours = np.ones((data.shape[0],), dtype=np.int32) * -1

        for _ in enumerate(nodes):
            selected_node = np.nan

            for node in nodes:
                if not is_node_used[node] and (np.isnan(selected_node) or min_weight[node] < min_weight[selected_node]):
                    selected_node = node

            assert not np.isnan(selected_node)
            if best_neighbours[selected_node] != -1:
                forest.add_edge(selected_node, best_neighbours[selected_node], min_weight[selected_node])
            is_node_used[selected_node] = True

            SpanningTreeClustering.__reweigh_nodes(data, selected_node, min_weight, best_neighbours)

        return forest

    @staticmethod
    @njit(cache=True, nogil=True, parallel=True)
    def __reweigh_nodes(data, selected_node, min_weight, best_neighbours):
        for node in prange(data.shape[0]):
            edge_weight = np.linalg.norm(data[selected_node] - data[node])
            if edge_weight < min_weight[node]:
                min_weight[node] = edge_weight
                best_neighbours[node] = selected_node

    def __get_cluster_params(self, spanning_forest: SpanningForest, cluster_idx: int) -> (np.ndarray, list, np.ndarray):
        root = spanning_forest.get_roots()[cluster_idx]
        cluster_edges = spanning_forest.get_edges(root)

        if len(cluster_edges) == 0:
            cluster_ids = np.array([root])
            cluster_center = self.__data[cluster_ids.squeeze()]
        else:
            cluster_ids = np.unique(list(map(lambda edge: [edge.first_node, edge.second_node], cluster_edges)))
            cluster_center = np.mean(self.__data[cluster_ids], axis=0)

        return cluster_ids, cluster_edges, cluster_center

    def __get_distance_matrix(self) -> np.ndarray:
        pool_args = (defaultdict(None, {
            "shared_data": RawArray(ctypes.c_double, self.__data.flatten()),
            "shared_partition": RawArray(ctypes.c_double, self.__partition.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, self.__data.shape[0]),
            "shared_clusters_count": RawValue(ctypes.c_int32, self.__partition.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exponent)
        }),)

        with ProcessPoolExecutor(max_workers=self.num_of_workers, initializer=pool_init, initargs=pool_args) as pool:
            distance_matrix = np.vstack(list(pool.map(parallel_compute_distances, range(self.__partition.shape[0]))))

        return distance_matrix

    def __check_first_criterion(self, edge_weight: float) -> bool:
        all_edges = self.__forest.get_edges(self.__forest.get_roots()[0])
        criterion = self.cutting_condition * sum(map(lambda edge: edge.weight, all_edges)) / (self.__data.shape[0] - 1)
        return edge_weight >= criterion

    def __check_second_criterion(self, edge_weights: np.ndarray, edge_index: int) -> bool:
        weight = edge_weights[edge_index]
        edge_weights = np.delete(edge_weights, edge_index)
        return weight / np.mean(edge_weights) >= self.cutting_condition

    def __use_third_criterion(self, cluster_edges: list, noise_roots: set) -> Edge:
        min_total_fhv = float("inf")
        bad_edge_index = -1

        forest = SpanningForest(self.__data.shape[0])
        for cluster_edge in cluster_edges:
            forest.add_edge(cluster_edge.first_node, cluster_edge.second_node, cluster_edge.weight)

        for edge_index, cluster_edge in enumerate(cluster_edges):
            forest.remove_edge(cluster_edge.first_node, cluster_edge.second_node)

            roots = forest.get_roots()

            left_root = forest.find_root(cluster_edge.first_node)
            left_cluster_ids, _, cluster_center = self.__get_cluster_params(forest, roots.index(left_root))
            left_fhv = fuzzy_hyper_volume(self.__data, self.weighting_exponent, left_cluster_ids, cluster_center)
            if left_fhv == -1:
                noise_roots.add(left_root)

            right_root = forest.find_root(cluster_edge.second_node)
            right_cluster_ids, _, cluster_center = self.__get_cluster_params(forest, roots.index(right_root))
            right_fhv = fuzzy_hyper_volume(self.__data, self.weighting_exponent, right_cluster_ids, cluster_center)
            if right_fhv == -1:
                noise_roots.add(right_root)

            if left_fhv != -1 and right_fhv != -1:
                if (total_fhv := left_fhv + right_fhv) <= min_total_fhv:
                    bad_edge_index = edge_index
                    min_total_fhv = total_fhv
            else:
                bad_edge_index = edge_index
                break

            forest.add_edge(cluster_edge.first_node, cluster_edge.second_node, cluster_edge.weight)

        return cluster_edges[bad_edge_index]

    def __simple_clustering(self):
        noise_roots = set()

        pool_args = (defaultdict(None, {
            "shared_data": RawArray(ctypes.c_double, self.__data.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, self.__data.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exponent)
        }),)

        with ProcessPoolExecutor(max_workers=self.num_of_workers, initializer=pool_init, initargs=pool_args) as pool:
            while self.__forest.size() < self.num_of_clusters:
                params = list(map(
                    lambda cluster: self.__get_cluster_params(self.__forest, cluster), range(self.__forest.size())
                ))
                futures = list(pool.submit(parallel_fuzzy_hyper_volume, ids, center) for ids, _, center in params)
                wait(futures, return_when=ALL_COMPLETED)
                fuzzy_volumes = np.fromiter(map(lambda future: future.result(), futures), dtype=np.float64)

                bad_cluster_edges = self.__forest.get_edges(self.__forest.get_roots()[np.argmax(fuzzy_volumes)])
                weights = np.fromiter(map(lambda edge: edge.weight, bad_cluster_edges), dtype=np.float64)
                max_weight_idx = int(np.argmax(weights))
                max_weight = weights[max_weight_idx]

                if self.__check_first_criterion(max_weight) or self.__check_second_criterion(weights, max_weight_idx):
                    worst_edge = bad_cluster_edges[max_weight_idx]
                else:
                    worst_edge = self.__use_third_criterion(bad_cluster_edges, noise_roots)

                self.__forest.remove_edge(worst_edge.first_node, worst_edge.second_node)

        self.__partition = np.zeros((self.__forest.size(), self.__data.shape[0]))

        roots = self.__forest.get_roots()
        for cluster in range(self.__forest.size()):
            cluster_ids, _, _ = self.__get_cluster_params(self.__forest, cluster)
            self.__partition[cluster, cluster_ids] = 1

            if roots[cluster] in noise_roots:
                self.__noise_occurrences.add(cluster)

        return self.__partition

    def __gath_geva_algorithm(self):
        previous_partition = np.zeros_like(self.__partition)

        while np.linalg.norm(self.__partition - previous_partition) > self.termination_tolerance:
            previous_partition = self.__partition.copy()
            power = Decimal(2 / (self.weighting_exponent - 1))

            distance_matrix = self.__get_distance_matrix()

            for cluster in np.arange(self.num_of_clusters):
                for point_index in np.arange(self.__partition.shape[1]):
                    distance = distance_matrix[cluster, point_index]
                    partition = sum(map(lambda other_cluster:
                                        (distance / distance_matrix[other_cluster, point_index]) ** power,
                                        np.arange(self.num_of_clusters)))
                    partition **= -1
                    self.__partition[cluster, point_index] = partition


# Parallel computation.

shared_data: RawArray
shared_partition: RawArray
shared_rows_count: RawValue
shared_clusters_count: RawValue
shared_weighting_exponent: RawValue


def pool_init(pool_args: defaultdict):
    global shared_data, shared_partition, shared_rows_count, shared_clusters_count, shared_weighting_exponent

    shared_data = pool_args.get("shared_data")
    shared_partition = pool_args.get("shared_partition")
    shared_rows_count = pool_args.get("shared_rows_count")
    shared_clusters_count = pool_args.get("shared_clusters_count")
    shared_weighting_exponent = pool_args.get("shared_weighting_exponent")


def parallel_fuzzy_hyper_volume(cluster_ids: np.ndarray, cluster_center: np.ndarray) -> float:
    data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
    weighting_exponent = shared_weighting_exponent.value
    return fuzzy_hyper_volume(data, weighting_exponent, cluster_ids, cluster_center)


def parallel_compute_distances(cluster: int) -> np.ndarray:
    data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
    weighting_exponent = shared_weighting_exponent.value
    partition = np.frombuffer(shared_partition).reshape((shared_clusters_count.value, -1))
    return cluster_distances(data, weighting_exponent, partition, cluster)

import ctypes
import numpy as np

from spanning_tree_clustering.clustering_utils import fuzzy_hyper_volume, cluster_distances
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from spanning_tree_clustering.cpp_utils import SpanningForest, Edge
from multiprocessing.sharedctypes import RawArray, RawValue
from sklearn.preprocessing import normalize
from numba import njit, prange
from decimal import Decimal

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class SpanningTreeClustering(object):
    cutting_condition: float
    weighting_exponent: float
    termination_tolerance: float
    num_of_workers: int
    clustering_algorithm: str

    _data: np.ndarray
    _forest: SpanningForest
    _partition: np.ndarray
    _noise: set

    def __init__(self, cutting_cond: float, weighting_exp: float, termination_tolerance: float,
                 num_of_workers: int = 1, clustering_algorithm: str = "hybrid"):
        self.cutting_condition = cutting_cond
        self.weighting_exponent = weighting_exp
        self.termination_tolerance = termination_tolerance
        self.num_of_workers = num_of_workers
        self.clustering_algorithm = clustering_algorithm

    def fit(self, data: np.ndarray, num_of_clusters):
        self.num_of_clusters = num_of_clusters

        self._data = normalize(data)
        self._forest = SpanningTreeClustering.build_mst(self._data)
        self._noise = set()

        if self.clustering_algorithm == "simple" or self.clustering_algorithm == "hybrid":
            self._partition = self.__simple_clustering()

        if self.clustering_algorithm == "gath-geva" or self.clustering_algorithm == "hybrid":
            # TODO: fix method.
            self._partition = self.__gath_geva_algorithm()

        return self

    @property
    def labels(self) -> np.ndarray:
        labels = np.argmax(self._partition, axis=0)
        labels_with_noise = np.where(labels == self._noise, -1, labels)
        return labels_with_noise

    @property
    def partition(self) -> np.ndarray:
        return self._partition

    @property
    def clusters_count(self) -> np.ndarray:
        return self._partition.shape[0]

    @staticmethod
    def build_mst(data: np.ndarray) -> SpanningForest:
        forest = SpanningForest(data.shape[0])

        nodes = np.arange(data.shape[0])
        is_node_used = np.zeros((data.shape[0],), dtype=np.bool)
        min_weight = np.ones((data.shape[0],), dtype=np.float64) * np.inf
        min_weight[0] = 0
        best_neighbours = np.ones((data.shape[0],), dtype=np.int32) * -1

        # TODO: Looks not good. Try to replace it with cython.
        @njit(cache=True, nogil=True, parallel=True)
        def reweigh_nodes(data_local, selected_node_local, min_weights_local, best_neighbours_local):
            for node_local in prange(data_local.shape[0]):
                edge_weight = np.linalg.norm(data_local[selected_node_local] - data_local[node_local])
                if edge_weight < min_weights_local[node_local]:
                    min_weights_local[node_local] = edge_weight
                    best_neighbours_local[node_local] = selected_node_local

        for _ in enumerate(nodes):
            selected_node = np.nan

            for node in nodes:
                if not is_node_used[node] and (np.isnan(selected_node) or min_weight[node] < min_weight[selected_node]):
                    selected_node = node

            assert not np.isnan(selected_node)
            if best_neighbours[selected_node] != -1:
                forest.add_edge(selected_node, best_neighbours[selected_node], min_weight[selected_node])
            is_node_used[selected_node] = True

            reweigh_nodes(data, selected_node, min_weight, best_neighbours)

        assert forest.is_spanning_tree()

        return forest

    def _get_cluster_params(self, spanning_forest: SpanningForest, cluster_idx: int) -> (np.ndarray, list, np.ndarray):
        root = spanning_forest.get_roots()[cluster_idx]
        cluster_edges = spanning_forest.get_edges(root)

        if len(cluster_edges) == 0:
            cluster_ids = np.array([root])
            cluster_center = self._data[cluster_ids.squeeze()]
        else:
            cluster_ids = np.unique(list(map(lambda edge: [edge.first_node, edge.second_node], cluster_edges)))
            cluster_center = np.mean(self._data[cluster_ids], axis=0)

        return cluster_ids, cluster_edges, cluster_center

    def _get_distance_matrix(self) -> np.ndarray:
        pool_args = (dict({
            "shared_data": RawArray(ctypes.c_double, self._data.flatten()),
            "shared_partition": RawArray(ctypes.c_double, self._partition.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, self._data.shape[0]),
            "shared_clusters_count": RawValue(ctypes.c_int32, self._partition.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exponent)
        }),)

        with ProcessPoolExecutor(max_workers=self.num_of_workers, initializer=pool_init, initargs=pool_args) as pool:
            distance_matrix = np.vstack(list(pool.map(parallel_compute_distances, range(self._partition.shape[0]))))

        return distance_matrix

    def _check_first_criterion(self, edge_weight: float) -> bool:
        all_edges = self._forest.get_edges(self._forest.get_roots()[0])
        criterion = self.cutting_condition * sum(map(lambda edge: edge.weight, all_edges)) / (self._data.shape[0] - 1)
        return edge_weight >= criterion

    def _check_second_criterion(self, edge_weights: np.ndarray, edge_index: int) -> bool:
        weight = edge_weights[edge_index]
        edge_weights = np.delete(edge_weights, edge_index)
        return weight / np.mean(edge_weights) >= self.cutting_condition

    def _use_third_criterion(self, cluster_edges: list, noise_roots: set) -> Edge:
        min_total_fhv = float("inf")
        bad_edge_index = -1

        forest = SpanningForest(self._data.shape[0])
        for cluster_edge in cluster_edges:
            forest.add_edge(cluster_edge.first_node, cluster_edge.second_node, cluster_edge.weight)

        for edge_index, cluster_edge in enumerate(cluster_edges):
            forest.remove_edge(cluster_edge.first_node, cluster_edge.second_node)

            roots = forest.get_roots()

            left_root = forest.find_root(cluster_edge.first_node)
            left_cluster_ids, _, cluster_center = self._get_cluster_params(forest, roots.index(left_root))
            left_fhv = fuzzy_hyper_volume(self._data, self.weighting_exponent, left_cluster_ids, cluster_center)
            if left_fhv == -1:
                noise_roots.add(left_root)

            right_root = forest.find_root(cluster_edge.second_node)
            right_cluster_ids, _, cluster_center = self._get_cluster_params(forest, roots.index(right_root))
            right_fhv = fuzzy_hyper_volume(self._data, self.weighting_exponent, right_cluster_ids, cluster_center)
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

        pool_args = (dict({
            "shared_data": RawArray(ctypes.c_double, self._data.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, self._data.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exponent)
        }),)

        with ProcessPoolExecutor(max_workers=self.num_of_workers, initializer=pool_init, initargs=pool_args) as pool:
            while self._forest.size() < self.num_of_clusters:
                params = list(map(lambda c: self._get_cluster_params(self._forest, c), range(self._forest.size())))
                futures = list(pool.submit(parallel_fuzzy_hyper_volume, ids, center) for ids, _, center in params)
                wait(futures, return_when=ALL_COMPLETED)
                fuzzy_volumes = np.fromiter(map(lambda future: future.result(), futures), dtype=np.float64)

                bad_cluster_edges = self._forest.get_edges(self._forest.get_roots()[np.argmax(fuzzy_volumes)])
                weights = np.fromiter(map(lambda edge: edge.weight, bad_cluster_edges), dtype=np.float64)
                max_weight_idx = int(np.argmax(weights))
                max_weight = weights[max_weight_idx]

                if self._check_first_criterion(max_weight) or self._check_second_criterion(weights, max_weight_idx):
                    worst_edge = bad_cluster_edges[max_weight_idx]
                else:
                    worst_edge = self._use_third_criterion(bad_cluster_edges, noise_roots)

                self._forest.remove_edge(worst_edge.first_node, worst_edge.second_node)

        self._partition = np.zeros((self._forest.size(), self._data.shape[0]))

        roots = self._forest.get_roots()
        for cluster in range(self._forest.size()):
            cluster_ids, _, _ = self._get_cluster_params(self._forest, cluster)
            self._partition[cluster, cluster_ids] = 1

            if roots[cluster] in noise_roots:
                self._noise.add(cluster)

        return self._partition

    def __gath_geva_algorithm(self):
        previous_partition = np.zeros_like(self._partition)

        while np.linalg.norm(self._partition - previous_partition) > self.termination_tolerance:
            previous_partition = self._partition.copy()
            power = Decimal(2 / (self.weighting_exponent - 1))

            distance_matrix = self._get_distance_matrix()

            for cluster in np.arange(self.num_of_clusters):
                for point_index in np.arange(self._partition.shape[1]):
                    distance = distance_matrix[cluster, point_index]
                    partition = sum(map(lambda other_cluster:
                                        (distance / distance_matrix[other_cluster, point_index]) ** power,
                                        np.arange(self.num_of_clusters)))
                    partition **= -1
                    self._partition[cluster, point_index] = partition


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

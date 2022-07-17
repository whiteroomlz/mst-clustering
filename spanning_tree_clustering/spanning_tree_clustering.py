import ctypes
import numpy as np

from decimal import Decimal

from multiprocessing.sharedctypes import RawArray, RawValue
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

from scipy.spatial import KDTree, distance_matrix as distances
from sklearn.preprocessing import normalize
from scipy.special import softmax

from spanning_tree_clustering.cpp_utils import DSU, SpanningForest, Edge
from spanning_tree_clustering.clustering_utils import fuzzy_hyper_volume, cluster_distances

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


# TODO: Требует тестирования и оптимизации кода как на c++, так и на Python.
class SpanningTreeClustering(object):
    workers: int
    num_of_clusters: int
    cutting_condition: float
    exp: float
    termination_tolerance: float

    __data: np.ndarray
    __spanning_forest: SpanningForest
    __partition_matrix: np.ndarray
    __noise: set

    def __init__(self, num_of_workers=1):
        self.workers = num_of_workers

    def clustering(self, data, num_of_clusters, cutting_condition, weighting_exponent, termination_tolerance,
                   mst_algorithm="Prim", clustering_mode="hybrid"):
        if data.shape[0] <= 1:
            raise ValueError("Count of clustering values should be greater than 1.")

        self.num_of_clusters = num_of_clusters
        self.cutting_condition = cutting_condition
        self.exp = weighting_exponent
        self.termination_tolerance = termination_tolerance

        self.__data = normalize(data)
        self.__spanning_forest = SpanningForest()
        self.__partition_matrix = softmax(np.random.rand(num_of_clusters, self.__data.shape[0]), axis=0)
        self.__noise = set()

        if mst_algorithm == "Prim":
            self.__prim_algorithm(data)
        elif mst_algorithm == "Kruskal":
            self.__kruskal_algorithm()
        else:
            raise ValueError("Unexpected minimum spanning tree algorithm: %s", mst_algorithm)

        if clustering_mode == "simple" or clustering_mode == "hybrid":
            self.__simple_clustering()

        if clustering_mode == "Gath-Geva" or clustering_mode == "hybrid":
            self.__gath_geva_algorithm()

    def get_labels(self):
        labels = np.argmax(self.__partition_matrix, axis=0)
        return np.where(labels == self.__noise, -1, labels)

    def __kruskal_algorithm(self):
        dists = np.triu(distances(self.__data, self.__data))
        sorted_edges = np.argsort(dists.ravel())

        nodes_count = self.__data.shape[0]
        dsu = DSU(nodes_count)

        for index in sorted_edges:
            if not dsu.is_singleton():
                row = index // nodes_count
                column = index % nodes_count
                distance = dists[row, column]

                if distance > 0 and dsu.find(row) != dsu.find(column):
                    dsu.unite(row, column)
                    self.__spanning_forest.add_edge(row, column, distance)
            else:
                break

    def __prim_algorithm(self, data: np.ndarray):
        vertexes = self.__data[0]
        vertexes_indexes = list([0])
        nodes_count = self.__data.shape[0]

        while len(vertexes_indexes) != nodes_count:
            temp_data = np.delete(data, vertexes_indexes, axis=0)
            normalized_temp_data = np.delete(self.__data, vertexes_indexes, axis=0)

            kdtree = KDTree(normalized_temp_data)
            dists, neighbors = kdtree.query(vertexes, k=1)
            sorted_edges = np.argsort(dists)

            first_vertex = vertexes_indexes[sorted_edges[0]]

            if isinstance(neighbors, np.ndarray):
                second_vertex = neighbors[sorted_edges][0]
            else:
                second_vertex = neighbors
            offset = 0
            while not np.all(data[second_vertex + offset] == temp_data[second_vertex]):
                offset += 1
            second_vertex += offset

            if isinstance(dists, np.ndarray):
                distance = dists[sorted_edges][0]
            else:
                distance = dists

            self.__spanning_forest.add_edge(first_vertex, second_vertex, distance)
            vertexes = np.vstack((vertexes, self.__data[second_vertex]))
            vertexes_indexes.append(second_vertex)

    def __get_cluster_params(self, spanning_forest, cluster: int) -> (np.ndarray, list, np.ndarray):
        cluster_edges = spanning_forest.get_root_edges(cluster)

        if len(cluster_edges) == 0:
            cluster_ids = np.array([spanning_forest.get_root_id(cluster)])
            cluster_center = self.__data[cluster_ids.squeeze()]
        else:
            cluster_ids = np.unique(list(map(lambda edge: [edge.parent_id, edge.child_id], cluster_edges)))
            cluster_center = np.mean(self.__data[cluster_ids], axis=0)

        return cluster_ids, cluster_edges, cluster_center

    def __get_distance_matrix(self):
        shared_data_ = RawArray(ctypes.c_double, self.__data.flatten())
        shared_rows_count_ = RawValue(ctypes.c_int32, self.__data.shape[0])
        shared_weighting_exponent_ = RawValue(ctypes.c_double, self.exp)
        shared_partition_matrix_ = RawArray(ctypes.c_double, self.__partition_matrix.flatten())
        shared_clusters_count_ = RawValue(ctypes.c_int32, self.__partition_matrix.shape[0])
        pool_args = (shared_data_, shared_rows_count_, shared_weighting_exponent_, shared_partition_matrix_,
                     shared_clusters_count_)

        with ProcessPoolExecutor(max_workers=self.workers, initializer=pool_init, initargs=pool_args) as executor:
            distance_matrix = np.vstack(list(
                executor.map(parallel_compute_distances, range(self.__partition_matrix.shape[0]))
            ))

        return distance_matrix

    def __get_first_criterion(self) -> float:
        if self.__spanning_forest.size() != 1:
            raise ValueError("There should be an MST in the spanning forest.")

        all_edges = self.__spanning_forest.get_root_edges(0)
        return self.cutting_condition * sum(map(lambda edge: edge.weight, all_edges)) / (self.__data.shape[0] - 1)

    def __check_second_criterion(self, weights, edge_index) -> bool:
        weight = weights[edge_index]
        weights = np.delete(weights, edge_index)

        return weight / np.mean(weights) >= self.cutting_condition

    def __simple_clustering(self):
        first_criterion = self.__get_first_criterion()
        noise_delegates = set()

        shared_data_ = RawArray(ctypes.c_double, self.__data.flatten())
        shared_rows_count_ = RawValue(ctypes.c_int32, self.__data.shape[0])
        shared_weighting_exponent_ = RawValue(ctypes.c_double, self.exp)
        pool_args = (shared_data_, shared_rows_count_, shared_weighting_exponent_,)

        with ProcessPoolExecutor(max_workers=self.workers, initializer=pool_init, initargs=pool_args) as executor:
            while self.__spanning_forest.size() < self.num_of_clusters:
                forest = self.__spanning_forest
                params = list(map(
                    lambda cluster_idx: self.__get_cluster_params(forest, cluster_idx), range(forest.size())
                ))

                futures = list(executor.submit(parallel_fuzzy_hyper_volume, ids, center) for ids, _, center in params)
                wait(futures, return_when=ALL_COMPLETED)
                fuzzy_volumes = np.fromiter(map(lambda future: future.result(), futures), dtype=np.float64)

                worst_cluster = np.argmax(fuzzy_volumes)
                cluster_edges = self.__spanning_forest.get_root_edges(worst_cluster)
                weights = list(map(lambda edge: edge.weight, cluster_edges))
                largest_weight_index = np.argmax(weights)
                largest_weight = weights[largest_weight_index]

                if largest_weight >= first_criterion or self.__check_second_criterion(weights, largest_weight_index):
                    bad_edge = cluster_edges[largest_weight_index]
                else:
                    temp_forest = SpanningForest()
                    for cluster_edge in cluster_edges:
                        temp_forest.add_edge(cluster_edge.parent_id, cluster_edge.child_id, cluster_edge.weight)

                    min_total_fhv = float("inf")
                    bad_edge_index = -1

                    for edge_index, cluster_edge in enumerate(cluster_edges):
                        temp_forest.remove_edge(cluster_edge.parent_id, cluster_edge.child_id)

                        left_cluster_ids, _, cluster_center = self.__get_cluster_params(temp_forest, 0)
                        left_fhv = fuzzy_hyper_volume(self.__data, self.exp, left_cluster_ids, cluster_center)
                        right_cluster_ids, _, cluster_center = self.__get_cluster_params(temp_forest, 1)
                        right_fhv = fuzzy_hyper_volume(self.__data, self.exp, right_cluster_ids, cluster_center)

                        if left_fhv != -1 and right_fhv != -1:
                            total_fhv = left_fhv + right_fhv
                        else:
                            if left_cluster_ids.size == 1 or right_cluster_ids.size == 1:
                                total_fhv = float("inf")
                            else:
                                if left_fhv == -1:
                                    noise_delegates.add(temp_forest.get_root_id(0))
                                else:
                                    noise_delegates.add(temp_forest.get_root_id(1))
                                bad_edge_index = edge_index
                                break

                        temp_forest.add_edge(cluster_edge.parent_id, cluster_edge.child_id, cluster_edge.weight)

                        if total_fhv <= min_total_fhv:
                            min_total_fhv = total_fhv
                            bad_edge_index = edge_index
                    bad_edge: Edge = cluster_edges[bad_edge_index]
                self.__spanning_forest.remove_edge(bad_edge.parent_id, bad_edge.child_id)

        for cluster in np.arange(self.__spanning_forest.size()):
            cluster_ids, _, _ = self.__get_cluster_params(self.__spanning_forest, cluster)

            delegate_id = self.__spanning_forest.get_root_id(cluster)
            if delegate_id in noise_delegates:
                self.__noise.add(cluster)

            self.__partition_matrix[cluster, cluster_ids] = 1
            self.__partition_matrix[self.__partition_matrix != 1] = 0

    def __gath_geva_algorithm(self):
        previous_partition = np.zeros_like(self.__partition_matrix)

        while np.linalg.norm(self.__partition_matrix - previous_partition) > self.termination_tolerance:
            previous_partition = self.__partition_matrix.copy()
            power = Decimal(2 / (self.exp - 1))

            distance_matrix = self.__get_distance_matrix()

            for cluster in np.arange(self.num_of_clusters):
                for point_index in np.arange(self.__partition_matrix.shape[1]):
                    distance = distance_matrix[cluster, point_index]
                    partition = sum(map(lambda other_cluster:
                                        (distance / distance_matrix[other_cluster, point_index]) ** power,
                                        np.arange(self.num_of_clusters)))
                    partition **= -1
                    self.__partition_matrix[cluster, point_index] = partition


def pool_init(data_, rows_count_, weighting_exponent_, partition_matrix_=None, clusters_count_=None):
    global shared_data, shared_rows_count, shared_weighting_exponent, shared_partition_matrix, shared_clusters_count
    shared_data = data_
    shared_rows_count = rows_count_
    shared_weighting_exponent = weighting_exponent_
    shared_partition_matrix = partition_matrix_
    shared_clusters_count = clusters_count_


def parallel_fuzzy_hyper_volume(cluster_ids: np.ndarray, cluster_center: np.ndarray) -> float:
    data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
    weighting_exponent = shared_weighting_exponent.value
    return fuzzy_hyper_volume(data, weighting_exponent, cluster_ids, cluster_center)


def parallel_compute_distances(cluster: int) -> np.ndarray:
    data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
    weighting_exponent = shared_weighting_exponent.value
    partition_matrix = np.frombuffer(shared_partition_matrix).reshape((shared_clusters_count.value, -1))
    return cluster_distances(data, weighting_exponent, partition_matrix, cluster)

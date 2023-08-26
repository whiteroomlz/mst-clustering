import math
import ctypes
import numpy as np

from itertools import product
from operator import itemgetter
from scipy.spatial import KDTree
from numpy import ndarray as ndarr
from abc import ABC, abstractmethod
from concurrent.futures import wait, ALL_COMPLETED

from mst_clustering.multiprocessing_tools import SharedMemoryPool, submittable
from multiprocessing.sharedctypes import RawArray, RawValue
from mst_clustering.cpp_adapters import SpanningForest


class ClusteringModel(ABC):
    @abstractmethod
    def __call__(self, data: ndarr, forest: SpanningForest, workers: int = 1, partition: ndarr = None) -> ndarr:
        pass

    @staticmethod
    def get_cluster_info(data: ndarr, forest: SpanningForest, cluster_idx: int) -> (ndarr, ndarr):
        root = forest.get_roots()[cluster_idx]
        cluster_ids = forest.get_tree_nodes(root)

        if len(cluster_ids) == 1:
            cluster_center = data[cluster_ids[0]]
        else:
            cluster_center = np.mean(data[cluster_ids], axis=0)

        return cluster_ids, cluster_center


class ZahnModel(ClusteringModel):
    cutting_cond: float
    hv_condition: float
    weighting_exp: float
    min_points_in_cluster: int
    num_of_clusters: int
    use_first_criterion: bool
    use_third_criterion: bool
    use_second_criterion: bool

    __kdtree: KDTree or None

    def __init__(
            self,
            cutting_condition=2.5,
            weighting_exponent=2,
            hv_condition=1e-4,
            min_points_in_cluster: int = 1,
            max_num_of_clusters: int = -1,
            use_first_criterion: bool = True,
            use_second_criterion: bool = True,
            use_third_criterion: bool = True
    ):
        self.cutting_cond = cutting_condition
        self.weighting_exp = weighting_exponent
        self.hv_condition = hv_condition
        self.min_points_in_cluster = min_points_in_cluster
        self.num_of_clusters = max_num_of_clusters
        self.use_first_criterion = use_first_criterion
        self.use_second_criterion = use_second_criterion
        self.use_third_criterion = use_third_criterion

        self.__kdtree = None

    def __call__(self, data: ndarr, forest: SpanningForest, workers: int = 1, partition: ndarr = None) -> ndarr:
        all_edges = forest.get_all_edges()

        shared_memory_dict = dict({
            "shared_data": RawArray(ctypes.c_double, data.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, data.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exp)
        })

        finished_clusters = set()
        with SharedMemoryPool(max_workers=workers, shared_memory_dict=shared_memory_dict) as pool:
            while self._check_num_of_clusters(forest):
                info = map(lambda c: ZahnModel.get_cluster_info(data, forest, c), range(forest.size))
                unfinished_clusters = list(filter(
                    lambda cluster_info: cluster_info[0].size >= self.min_points_in_cluster, info
                ))

                if len(unfinished_clusters) == 0:
                    break

                futures = list(pool.submit(
                    ZahnModel.__fuzzy_hyper_volume_task, ids, center
                ) for ids, center in unfinished_clusters)
                wait(futures, return_when=ALL_COMPLETED)

                volumes = np.fromiter(map(lambda future: future.result(), futures), dtype=np.float64)
                volumes_without_noise = np.where(volumes == math.inf, -1, volumes)
                volumes_without_noise[list(finished_clusters)] = -1
                bad_cluster = np.argmax(volumes_without_noise)

                cluster_ids, cluster_edges = forest.get_tree_info(forest.get_roots()[bad_cluster])
                cluster_edges = list(map(lambda edge: edge.nodes(), cluster_edges))

                bad_edge_found = False
                if self.use_first_criterion:
                    bad_edge_found, bad_edge = self._apply_first_criterion(data, all_edges, cluster_edges)
                if not bad_edge_found and self.use_second_criterion:
                    if self.__kdtree is None:
                        self.__kdtree = KDTree(data)
                    bad_edge_found, bad_edge = self._apply_second_criterion(data, all_edges, cluster_edges, workers)
                if not bad_edge_found and self.use_third_criterion:
                    bad_edge_found, bad_edge = self._apply_third_criterion(
                        data, all_edges, cluster_ids, cluster_edges, forest, pool
                    )
                if not bad_edge_found:
                    if len(finished_clusters) == forest.size:
                        break
                    else:
                        finished_clusters.add(bad_cluster)
                        continue

                finished_clusters.clear()
                forest.remove_edge(*bad_edge)
                all_edges.pop(bad_edge)

        partition = np.zeros((forest.size, data.shape[0]))
        for cluster in range(forest.size):
            root = forest.get_roots()[cluster]
            cluster_ids = forest.get_tree_nodes(root)
            partition[cluster, cluster_ids] = 1

        return partition

    def _check_num_of_clusters(self, forest: SpanningForest) -> bool:
        return self.num_of_clusters == -1 or forest.size < self.num_of_clusters

    def _apply_first_criterion(self, data: ndarr, all_edges: dict, cluster_edges: list) -> (bool, tuple):
        most_heavy_edge = max(cluster_edges, key=lambda edge: all_edges[edge])
        criterion = self.cutting_cond * np.sum(np.fromiter(all_edges.values(), dtype=np.float64)) / (data.shape[0] - 1)
        edge_weight = all_edges[most_heavy_edge]

        return edge_weight >= criterion, most_heavy_edge

    def _apply_second_criterion(self, data: ndarr, all_edges: dict, cluster_edges: list, workers: int) -> (bool, tuple):
        edges_weights = itemgetter(*cluster_edges)(all_edges)
        edges_weights = [edges_weights] if isinstance(edges_weights, (int, float)) else list(edges_weights)
        all_edges_nodes_pairs = np.array(list(all_edges.keys())).T
        sorted_indices = np.argsort(edges_weights)[::-1]
        for index in sorted_indices:
            points = data[cluster_edges[index], :]
            radius = edges_weights[index]

            neighbours = self.__kdtree.query_ball_point(x=points, r=radius, workers=workers, return_sorted=False)

            _, neighbours1_indices, _ = np.intersect1d(all_edges_nodes_pairs[0], neighbours[0], return_indices=True)
            filtered1 = all_edges_nodes_pairs[:, neighbours1_indices]
            _, neighbours2_indices, _ = np.intersect1d(filtered1[1], neighbours[1], return_indices=True)
            filtered2 = filtered1[:, neighbours2_indices]
            if filtered2.shape[1] <= 1:
                continue
            neighbours_edges_weights = itemgetter(*zip(*filtered2))(all_edges)

            criterion = self.cutting_cond * sum(neighbours_edges_weights) / (len(neighbours_edges_weights))
            current_edge_weight = edges_weights[index]
            if current_edge_weight >= criterion:
                return True, cluster_edges[index]

        return False, None

    def _apply_third_criterion(self, data: ndarr, all_edges: dict, cluster_ids: np.ndarray, cluster_edges: list,
                               forest: SpanningForest, pool: SharedMemoryPool) -> (bool, tuple):
        futures = list()
        cluster_center = np.mean(data[cluster_ids], axis=0)
        for edge_index, cluster_edge in enumerate(cluster_edges):
            first_node, second_node = cluster_edge
            forest.remove_edge(first_node, second_node)

            roots = forest.get_roots()
            left_root = forest.find_root(first_node)
            right_root = forest.find_root(second_node)

            left_tree_size = forest.get_tree_size(left_root)
            right_tree_size = forest.get_tree_size(right_root)
            if left_tree_size > right_tree_size:
                left_root, right_root = right_root, left_root
                left_tree_size, right_tree_size = right_tree_size, left_tree_size

            left_cluster_ids, left_center = self.get_cluster_info(data, forest, roots.index(left_root))
            futures.append(pool.submit(ZahnModel.__fuzzy_hyper_volume_task, left_cluster_ids, cluster_center))

            right_cluster_ids = np.setdiff1d(cluster_ids, left_cluster_ids, assume_unique=True)
            right_center = (cluster_center * cluster_ids.size - left_center * left_tree_size) / right_tree_size
            futures.append(pool.submit(ZahnModel.__fuzzy_hyper_volume_task, right_cluster_ids, right_center))

            forest.add_edge(first_node, second_node, all_edges[(first_node, second_node)])
        wait(futures, return_when=ALL_COMPLETED)

        bad_edge_index = 0
        min_total_hv = math.inf
        for edge_index in range(len(cluster_edges)):
            left_hv = futures[2 * edge_index].result()
            right_hv = futures[2 * edge_index + 1].result()
            if not (left_hv == math.inf or right_hv == math.inf):
                total_hv = left_hv + right_hv
                if total_hv <= min_total_hv:
                    bad_edge_index = edge_index
                    min_total_hv = total_hv

        return min_total_hv > self.hv_condition and min_total_hv != math.inf, cluster_edges[bad_edge_index]

    @staticmethod
    @submittable
    def __fuzzy_hyper_volume_task(cluster_ids: ndarr, cluster_center: ndarr) -> float:
        # noinspection PyShadowingNames
        import numpy as np
        from mst_clustering.math_utils import hyper_volume

        shared_memory = SharedMemoryPool.get_shared_memory()
        shared_data = shared_memory["shared_data"]
        shared_rows_count = shared_memory["shared_rows_count"]
        shared_weighting_exponent = shared_memory["shared_weighting_exponent"]

        data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
        weighting_exponent = shared_weighting_exponent.value
        volume = hyper_volume(data, weighting_exponent, cluster_ids, cluster_center)

        return volume


class GathGevaModel(ClusteringModel):
    termination_tolerance: float
    weighting_exp: float

    def __init__(self, termination_tolerance: float = 1e-4, weighting_exponent: float = 2):
        self.termination_tolerance = termination_tolerance
        self.weighting_exp = weighting_exponent

    def __call__(self, data: ndarr, forest: SpanningForest, workers: int = 1, partition: ndarr = None) -> ndarr:
        assert partition is not None, "This clustering method requires a non None partition matrix."

        non_noise = ~np.all(partition == 0, axis=1)
        non_noise_clusters = np.arange(partition.shape[0])[non_noise]

        while True:
            previous_partition = partition.copy()
            power = 2 / (self.weighting_exp - 1)

            ln_distance_matrix = self._get_ln_distance_matrix(data, partition, non_noise_clusters, workers)

            for cluster, point_idx in product(non_noise_clusters, np.arange(partition.shape[1])):
                ln_distance = ln_distance_matrix[cluster, point_idx]

                new_partition = 0.0
                for other_cluster in non_noise_clusters:
                    new_partition += np.exp((ln_distance - ln_distance_matrix[other_cluster, point_idx]) * power)
                new_partition **= -1

                partition[cluster, point_idx] = new_partition

            partitions_distance = np.linalg.norm(partition - previous_partition)
            if partitions_distance < self.termination_tolerance:
                break

        return partition

    def _get_ln_distance_matrix(self, data: ndarr, partition: ndarr, non_noise_clusters: ndarr,
                                workers: int) -> ndarr:
        shared_memory_dict = dict({
            "shared_data": RawArray(ctypes.c_double, data.flatten()),
            "shared_partition": RawArray(ctypes.c_double, partition.flatten()),
            "shared_rows_count": RawValue(ctypes.c_int32, data.shape[0]),
            "shared_clusters_count": RawValue(ctypes.c_int32, partition.shape[0]),
            "shared_weighting_exponent": RawValue(ctypes.c_double, self.weighting_exp)
        })

        with SharedMemoryPool(max_workers=workers, shared_memory_dict=shared_memory_dict) as pool:
            futures = [pool.submit(GathGevaModel.__compute_distances_task, cluster) for cluster in non_noise_clusters]

            wait(futures, return_when=ALL_COMPLETED)

            distance_matrix = np.zeros_like(partition)
            distance_matrix[non_noise_clusters] = list(map(lambda future: future.result(), futures))

        return distance_matrix

    @staticmethod
    @submittable
    def __compute_distances_task(cluster: int) -> ndarr:
        # noinspection PyShadowingNames
        import numpy as np
        from mst_clustering.math_utils import cluster_ln_distances

        shared_memory = SharedMemoryPool.get_shared_memory()
        shared_data = shared_memory["shared_data"]
        shared_partition = shared_memory["shared_partition"]
        shared_rows_count = shared_memory["shared_rows_count"]
        shared_clusters_count = shared_memory["shared_clusters_count"]
        shared_weighting_exponent = shared_memory["shared_weighting_exponent"]

        data = np.frombuffer(shared_data).reshape((shared_rows_count.value, -1))
        weighting_exponent = shared_weighting_exponent.value
        partition = np.frombuffer(shared_partition).reshape((shared_clusters_count.value, -1))
        ln_distances = cluster_ln_distances(data, weighting_exponent, partition, cluster)

        return ln_distances

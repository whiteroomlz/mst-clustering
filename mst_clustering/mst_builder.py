import numpy as np

from abc import ABC
from numba import njit, prange

from mst_clustering.cpp_adapters import SpanningForest


class MstBuilder(ABC):
    @staticmethod
    def build(data: np.ndarray) -> SpanningForest:
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

        assert forest.is_spanning_tree

        return forest

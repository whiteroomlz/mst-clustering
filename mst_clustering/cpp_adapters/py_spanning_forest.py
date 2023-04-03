import mst_lib
import numpy as np


class Edge(object):
    first_node: int
    second_node: int
    weight: float

    def __init__(self, spanning_forest_edge: mst_lib.Edge):
        self.first_node = spanning_forest_edge.first_node
        self.second_node = spanning_forest_edge.second_node
        self.weight = spanning_forest_edge.edge_weight

    @property
    def nodes(self) -> tuple:
        return self.first_node, self.second_node


class SpanningForest(object):
    __spanning_forest: mst_lib.SpanningForest

    def __init__(self, size: int = 0, spanning_forest: mst_lib.SpanningForest = None):
        if spanning_forest is not None:
            self.__spanning_forest = spanning_forest
        else:
            self.__spanning_forest = mst_lib.SpanningForest(size)

    @property
    def is_spanning_tree(self) -> bool:
        return self.__spanning_forest.is_spanning_tree()

    @property
    def size(self) -> int:
        return self.__spanning_forest.size()

    def find_root(self, node) -> int:
        return self.__spanning_forest.find_root(node)

    def get_tree_size(self, root) -> int:
        return self.__spanning_forest.get_tree_size(root)

    def add_edge(self, first_node: int, second_node: int, weight: float) -> None:
        self.__spanning_forest.add_edge(first_node, second_node, weight)

    def remove_edge(self, first_node: int, second_node: int) -> None:
        self.__spanning_forest.remove_edge(first_node, second_node)

    def get_roots(self) -> list:
        roots = mst_lib.Int32Vector()
        self.__spanning_forest.get_roots(roots)
        return list(roots)

    def get_tree_info(self, root) -> (np.ndarray, mst_lib.EdgeVector):
        nodes, edges = self.__spanning_forest.get_tree_info(root)
        return nodes, edges

    def get_tree_nodes(self, root) -> np.ndarray:
        nodes = self.__spanning_forest.get_tree_nodes(root)
        return nodes

    def get_tree_edges(self, root) -> mst_lib.EdgeVector:
        edges = self.__spanning_forest.get_tree_edges(root)
        return edges

    def get_all_edges(self) -> dict:
        all_edges = dict()
        for root in self.get_roots():
            for edge in self.get_tree_edges(root):
                all_edges[edge.nodes()] = edge.weight
        return all_edges

    def save(self, filename):
        all_edges = list()
        for root in self.get_roots():
            all_edges.extend(
                list(map(lambda edge: [edge.first_node, edge.second_node, edge.weight], self.get_tree_info(root)[1])))
        np.save(filename, np.array(all_edges))

    @staticmethod
    def load(filename):
        all_edges = np.load(filename)
        tree_size = int(max(np.max(all_edges[:, 0]), np.max(all_edges[:, 1])) + 1)
        spanning_forest = mst_lib.SpanningForest(tree_size)
        for edge in all_edges:
            spanning_forest.add_edge(int(edge[0]), int(edge[1]), edge[2])
        return SpanningForest(spanning_forest=spanning_forest)

import mst_lib


class Edge(object):
    first_node: int
    second_node: int
    weight: float

    def __init__(self, spanning_forest_edge: mst_lib.Edge):
        self.first_node = spanning_forest_edge.first_node
        self.second_node = spanning_forest_edge.second_node
        self.weight = spanning_forest_edge.edge_weight


class SpanningForest(object):
    __spanning_forest: mst_lib.SpanningForest

    def __init__(self, size: int = 0, spanning_forest: mst_lib.SpanningForest = None):
        if spanning_forest is None:
            self.__spanning_forest = mst_lib.SpanningForest(size)
        else:
            self.__spanning_forest = spanning_forest

    @property
    def is_spanning_tree(self) -> bool:
        return self.__spanning_forest.is_spanning_tree()

    @property
    def size(self) -> int:
        return self.__spanning_forest.size()

    def find_root(self, node) -> int:
        return self.__spanning_forest.find_root(node)

    def add_edge(self, first_node: int, second_node: int, weight: float) -> None:
        self.__spanning_forest.add_edge(first_node, second_node, weight)

    def remove_edge(self, first_node: int, second_node: int) -> None:
        self.__spanning_forest.remove_edge(first_node, second_node)

    def get_roots(self) -> list[int]:
        roots = list()
        self.__spanning_forest.get_roots(roots)
        return roots

    def get_edges(self, root) -> list[Edge]:
        edges = list()
        self.__spanning_forest.get_edges(root, edges)
        return list(map(lambda cpp_edge: Edge(cpp_edge), edges))

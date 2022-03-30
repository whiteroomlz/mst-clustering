import spanning_forest


class Edge(object):
    parent_id: int
    child_id: int
    weight: float

    def __init__(self, spanning_forest_edge: spanning_forest.Edge):
        self.parent_id = spanning_forest_edge.get_parent_id()
        self.child_id = spanning_forest_edge.get_child_id()
        self.weight = spanning_forest_edge.get_weight()


class SpanningForest(object):
    __spanning_forest_forest: spanning_forest.SpanningForest

    def __init__(self):
        self.__spanning_forest_forest = spanning_forest.SpanningForest()

    def size(self):
        return self.__spanning_forest_forest.size()

    def add_edge(self, first_id: int, second_id: int, weight: float):
        self.__spanning_forest_forest.add_edge(first_id, second_id, weight)

    def remove_edge(self, first_id: int, second_id: int):
        self.__spanning_forest_forest.remove_edge(first_id, second_id)

    def get_root_edges(self, index):
        edges = list()
        self.__spanning_forest_forest.get_root_edges(index, edges)

        return list(map(lambda edge: Edge(edge), edges))

    def get_root_id(self, index):
        return self.__spanning_forest_forest.get_root_id(index)

import mst_lib

from mst_clustering.cpp_adapters.py_spanning_forest import SpanningForest
from typing import List
from enum import Enum


class DistanceMeasure(Enum):
    COSINE = mst_lib.MstBuilder.COSINE
    EUCLIDEAN = mst_lib.MstBuilder.EUCLIDEAN
    QUADRATIC = mst_lib.MstBuilder.QUADRATIC


class MstBuilder(object):
    __mst_builder: mst_lib.MstBuilder

    def __init__(self, points: List[List]):
        self.__mst_builder = mst_lib.MstBuilder(points)

    def build(self, workers_count: int, distance_measure: DistanceMeasure) -> SpanningForest:
        return SpanningForest(spanning_forest=self.__mst_builder.build(workers_count, distance_measure.value))

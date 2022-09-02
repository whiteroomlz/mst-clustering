import copy
import numpy as np

from typing import Iterator
from sklearn.preprocessing import normalize

from mst_clustering.clustering_models import ClusteringModel
from mst_clustering.cpp_adapters import SpanningForest
from mst_clustering.mst_builder import MstBuilder

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class Pipeline:
    clustering_models: list[ClusteringModel]
    spanning_forest: SpanningForest = None
    partition: np.ndarray

    __models_iterator: Iterator[ClusteringModel]
    __data: np.ndarray

    def __init__(self, clustering_models: list[ClusteringModel]):
        self.clustering_models = clustering_models.copy()
        self.__models_iterator = iter(clustering_models)

    def fit(self, data: np.ndarray = None, workers_count: int = 1, initial_partition: np.ndarray = None,
            spanning_forest: SpanningForest = None, n_steps: int = None, use_normalization: bool = True):
        if data is not None:
            self.__data = normalize(data.copy()) if use_normalization else data.copy()

        if spanning_forest is not None:
            self.spanning_forest = copy.deepcopy(spanning_forest)
        elif self.spanning_forest is None:
            self.spanning_forest = MstBuilder.build(data)

        partition = initial_partition

        for _ in range(n_steps if n_steps is not None else len(self.clustering_models)):
            try:
                model = next(self.__models_iterator)
            except StopIteration:
                break

            partition = model(data=self.__data, forest=self.spanning_forest, workers=workers_count, partition=partition)
            self.partition = partition

        return self

    @property
    def labels(self) -> np.ndarray:
        labels = np.argmax(self.partition, axis=0)
        return labels

    @property
    def clusters_count(self) -> np.ndarray:
        return self.partition.shape[0]

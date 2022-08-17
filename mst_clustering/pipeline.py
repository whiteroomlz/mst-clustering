import numpy as np

from mst_clustering.clustering_models import ClusteringModel
from mst_clustering.mst_builder import MstBuilder
from sklearn.preprocessing import normalize

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class Pipeline:
    clustering_models: list[ClusteringModel]

    _partition: np.ndarray

    def __init__(self, clustering_models: list[ClusteringModel]):
        self.clustering_models = clustering_models.copy()

    def fit(self, data: np.ndarray, workers_count: int, initial_partition: np.ndarray = None):
        data = normalize(data.copy())
        spanning_forest = MstBuilder.build(data)

        partition = initial_partition
        for model in self.clustering_models:
            partition = model(data=data, forest=spanning_forest, workers=workers_count, partition=partition)
        self._partition = partition

        return self

    @property
    def labels(self) -> np.ndarray:
        labels = np.argmax(self._partition, axis=0)
        return labels

    @property
    def partition(self) -> np.ndarray:
        return self._partition

    @property
    def clusters_count(self) -> np.ndarray:
        return self._partition.shape[0]

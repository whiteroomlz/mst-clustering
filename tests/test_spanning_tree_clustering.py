import math
import multiprocessing

from mst_clustering.clustering_models import ZahnModel
from sklearn.datasets import make_blobs
from mst_clustering import Pipeline


if __name__ == "__main__":
    multiprocessing.freeze_support()

    X, y = make_blobs(n_samples=1000, n_features=10, centers=7)

    clustering = Pipeline(clustering_models=[
        ZahnModel(3, 1.5, math.inf, max_num_of_clusters=7, use_additional_criterion=False),
    ])
    clustering.fit(data=X, workers_count=4)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count

import numpy as np
import multiprocessing

from mst_clustering.clustering_models import ZahnModel
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from mst_clustering import Pipeline

scaler = StandardScaler()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    X, y = make_blobs(n_samples=1000, n_features=10, centers=7)
    X = scaler.fit_transform(X)

    clustering = Pipeline(clustering_models=[
        ZahnModel(3, 1.5, 1e-2, max_num_of_clusters=-1),
    ])
    clustering.fit(data=X, workers_count=4)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count

import multiprocessing
from unittest import TestCase
from sklearn.datasets import make_blobs

from mst_clustering.clustering_models import ZahnModel, GathGevaModel
from mst_clustering import Pipeline


class TestSpanningTreeClustering(TestCase):
    multiprocessing.freeze_support()

    X, y = make_blobs(n_samples=[800, 600, 1650], n_features=2, centers=[(-8, -4), (-4, 2), (-1, -2)], random_state=8)

    clustering = Pipeline(clustering_models=[
        ZahnModel(num_of_clusters=50, hv_condition=1e-4, use_additional_criterion=False),
        GathGevaModel(termination_tolerance=1e-3)
    ])
    clustering.fit(data=X, workers_count=6)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count
    noise = clustering.noise

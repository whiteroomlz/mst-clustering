import multiprocessing
from spanning_tree_clustering import SpanningTreeClustering
from sklearn.datasets import make_blobs
from unittest import TestCase
import matplotlib.pyplot as plt

import timeit


class TestSpanningTreeClustering(TestCase):
    multiprocessing.freeze_support()

    X, y = make_blobs(n_samples=1000, n_features=10, centers=7)

    clustering = SpanningTreeClustering(3, 1, 1, num_of_workers=6, clustering_algorithm="simple")
    clustering.fit(X, 7)
    labels = clustering.labels

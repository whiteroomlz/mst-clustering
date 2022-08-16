import multiprocessing
from spanning_tree_clustering import SpanningTreeClustering
from sklearn.datasets import make_blobs
from unittest import TestCase
import matplotlib.pyplot as plt

import timeit

class TestSpanningTreeClustering(TestCase):
    cls = SpanningTreeClustering(3, 1, 1, num_of_workers=6, clustering_algorithm="simple")

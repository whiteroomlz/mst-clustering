# Spanning tree clustering

## Description

This repository provides the Python package for clustering numpy arrays of n-dimensional vectors with methods based on a
spanning tree construction, such as the Gath-Geva clustering algorithm.

## Installation and usage

For installation use the next `pip` command:

```bash
    pip install git+https://github.com/whiteroomlz/spanning-tree-clustering.git
```

The class `SpanningTreeClustering` uses the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
module, so you should create an entry point in your main script:

```python
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ...
```

Usage example:

```python
import multiprocessing
from sklearn.datasets import make_blobs
from spanning_tree_clustering import SpanningTreeClustering

if __name__ == "__main__":
    multiprocessing.freeze_support()

    X, y = make_blobs(n_samples=1000, n_features=10, centers=7)

    clustering = SpanningTreeClustering(num_of_workers=4)
    clustering.fit(X, num_of_clusters=7, cutting_condition=3, termination_tolerance=1, weighting_exponent=1.5)
    labels = clustering.get_labels()
```

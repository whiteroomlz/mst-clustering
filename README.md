# Minimal Spanning Tree (MST) clustering

## Description

This repository provides the Python package for clustering numpy arrays of n-dimensional vectors with methods based on a
minimal spanning tree construction, such as the Zahn or Gath-Geva clustering algorithms.

## Installation and usage

For installation use the next `pip` command:

```bash
    pip install git+https://github.com/whiteroomlz/mst-clustering.git
```

The classes `ZahnModel` and `GathGevaModel` use the [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
module, so you should create an entry point in your main script if you include some of them into your `Pipeline`:

```python
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ...
```

Usage example:

```python
import numpy as np
import multiprocessing

from mst_clustering.clustering_models import ZahnModel
from sklearn.datasets import make_blobs
from mst_clustering import Pipeline


if __name__ == "__main__":
    multiprocessing.freeze_support()

    X, y = make_blobs(n_samples=1000, n_features=10, centers=7)

    clustering = Pipeline(clustering_models=[
        ZahnModel(3, 1.5, 1e-4, max_num_of_clusters=7),
    ])
    clustering.fit(data=X, workers_count=4)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count
```

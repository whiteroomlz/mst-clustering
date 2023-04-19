import multiprocessing

import pandas as pd

from mst_clustering.clustering_models import ZahnModel
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from mst_clustering import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.use('TkAgg')  # !IMPORTANT

scaler = StandardScaler()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # X, y = make_blobs(n_samples=1000, n_features=10, centers=7)
    X, y = list(make_blobs(n_samples=1000, centers=[(-5,-10), (-5,-5), (7,12), (7,5), (12,-12)],
                         random_state=8))
    # X = scaler.fit_transform(X)

    clustering = Pipeline(clustering_models=[
        ZahnModel(2.5, 1.5, 1e-2, max_num_of_clusters=50, use_third_criterion=False),
    ], fuzzy_noise_criterion=0.05)
    clustering.fit(data=X, workers_count=8)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clustering.labels, palette='deep')
    plt.show()

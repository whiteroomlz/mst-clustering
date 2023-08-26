import multiprocessing

from mst_clustering.clustering_models import ZahnModel
from sklearn.datasets import make_blobs
from mst_clustering import Pipeline

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.use('TkAgg')  # !IMPORTANT


if __name__ == "__main__":
    multiprocessing.freeze_support()

    # X, y = make_blobs(n_samples=1000, n_features=10, centers=7)
    X, y = list(make_blobs(n_samples=2, centers=[(-5, -10)], random_state=8))

    clustering = Pipeline(clustering_models=[
        ZahnModel(2.5, 1.5, 1e-2, max_num_of_clusters=50, use_third_criterion=False),
    ], fuzzy_noise_criterion=0.05)
    clustering.fit(data=X, workers_count=8)

    labels = clustering.labels
    partition = clustering.partition
    clusters_count = clustering.clusters_count

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=clustering.labels, palette='deep')
    plt.show()

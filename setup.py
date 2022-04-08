from setuptools import setup
from os.path import join, dirname

setup(
    name="spanning_tree_clustering",
    version='1.0',
    packages=["spanning_tree_clustering", "spanning_tree_clustering.cpp_utils"],
    install_requires=[
        "certifi",
        "joblib",
        "numpy",
        "scikit-learn",
        "scipy",
        "sklearn",
        "threadpoolctl",
        "wincertstore",
        "pybind11",
        "setuptools",
        "dsu @ git+https://github.com/Whiteroomlz/dsu.git",
        "spanning_forest @ git+https://github.com/Whiteroomlz/spanning_forest.git",
    ],
)

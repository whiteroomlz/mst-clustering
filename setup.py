import pybind11
from setuptools import setup, Extension

cpp_args = ['-std=c++2a', '-stdlib=libc++', '-mmacosx-version-min=10.7']

ext_modules = [
    Extension(
        'spanning_forest',
        sources=['spanning_tree_clustering/cpp_source/spanning_forest/spanning_forest.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
    Extension(
        'dsu',
        sources=['spanning_tree_clustering/cpp_source/dsu/dsu.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name="spanning_tree_clustering",
    version='1.1',
    description='Implementation of fuzzy clustering algorithms based on spanning trees',
    author='Nikita Borodin',
    author_email='borodinik.s@gmail.com',
    url='https://github.com/whiteroomlz/spanning-tree-clustering/',
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
    ],
    ext_modules=ext_modules,
)

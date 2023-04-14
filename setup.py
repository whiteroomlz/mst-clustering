import pybind11
from setuptools import setup, Extension

cpp_args = ['-std=c++2a']

ext_modules = [
    Extension(
        'mst_lib',
        sources=[
            'mst_clustering/cpp_source/mst_lib/spanning_forest.cpp',
            'mst_clustering/cpp_source/mst_lib/point.cpp',
            'mst_clustering/cpp_source/mst_lib/mst_builder.cpp',
            'mst_clustering/cpp_source/mst_lib/pybind11_compile.cpp',
        ],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name="mst_clustering",
    version='1.5',
    description='Implementation of fuzzy clustering algorithms based on spanning trees',
    author='Nikita Borodin',
    author_email='borodinik.s@gmail.com',
    url='https://github.com/whiteroomlz/mst-clustering/',
    packages=["mst_clustering", "mst_clustering.cpp_adapters"],
    install_requires=[
        "certifi",
        "joblib",
        "numpy",
        "numba",
        "scikit-learn",
        "scipy",
        "scikit-learn",
        "threadpoolctl",
        "wincertstore",
        "pybind11",
        "setuptools",
    ],
    ext_modules=ext_modules,
)

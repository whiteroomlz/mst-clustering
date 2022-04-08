from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++20', '-stdlib=libc++', '-mmacosx-version-min=10.7']

sfc_module = Extension(
    'spanning_forest',
    sources=['spanning_forest.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='spanning_forest',
    version='1.0',
    description='Python package with spanning_forest C++ extension (PyBind11)',
    ext_modules=[sfc_module],
)

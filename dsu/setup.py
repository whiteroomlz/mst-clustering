from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++20', '-stdlib=libc++', '-mmacosx-version-min=10.7']

sfc_module = Extension(
    'dsu',
    sources=['dsu.cpp'],
    include_dirs=[pybind11.get_include()],
    language='c++',
    extra_compile_args=cpp_args,
    )

setup(
    name='dsu',
    version='1.0',
    description='Python package with dsu C++ extension (PyBind11)',
    ext_modules=[sfc_module],
)

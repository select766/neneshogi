# Compile cython module:
# python setup.py build_ext --inplace
# Add to import path:
# python setup.py develop

from setuptools import setup, find_packages, Extension
import numpy as np


setup(
    name="neneshogi",
    version="1.0",
    packages=find_packages(),
    test_suite='test'
)

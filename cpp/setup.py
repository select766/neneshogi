# Add to import path:
# python setup.py develop

from setuptools import setup, find_packages, Extension

setup(
    name="neneshogi_cpp",
    version="1.0",
    packages=find_packages(),
    test_suite='test'
)

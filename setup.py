# Compile cython module:
# python setup.py build_ext --inplace
# Add to import path:
# python setup.py develop

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize, build_ext
import numpy as np

extensions = [
    Extension("*", ["neneshogi/*.pyx"],
              include_dirs=[np.get_include()])
]

setup(
    name="neneshogi",
    version="1.0",
    packages=find_packages(),
    test_suite='test',
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, annotate=True)
)

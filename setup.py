import setuptools
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


# Define the Cython extension
extension = Extension(
    name="carmapy.carmapy_c",
    sources=["carmapy/carmapy_c.pyx"],
    include_dirs=[numpy.get_include()]
)

# Use cythonize to build the extension
setup(
    name="carmapy",
    version='0.1.0',
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extension, language_level="3"),
)
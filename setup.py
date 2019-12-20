from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules=[
    Extension("louvain_cython", ["louvain_cython.pyx"], extra_compile_args = ["-ffast-math"]),
]


setup(
  ext_modules = cythonize(["*.pyx"]),
  )
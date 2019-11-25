from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

print("numpy .h path " + str(numpy.get_include()))

extensions = [
    Extension("edgemaze", ["edgemaze.pyx"],
          include_dirs=[numpy.get_include()],
          libraries=[numpy.get_include()],
          library_dirs=[numpy.get_include()])
    ]

setup(
    name="edgemaze cython",
    ext_modules=cythonize(extensions)
    )

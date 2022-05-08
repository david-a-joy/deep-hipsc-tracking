#!/usr/bin/env python3

import os

from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize

import numpy as np

basedir = os.path.dirname(os.path.realpath(__file__))
aboutfile = os.path.join(basedir, 'deep_hipsc_tracking', '__about__.py')
scriptdir = os.path.join(basedir, 'scripts')

# Load the info from the about file
about = {}
with open(aboutfile) as f:
    exec(f.read(), about)

scripts = [os.path.join('scripts', p)
           for p in os.listdir(scriptdir)
           if os.path.isfile(os.path.join(scriptdir, p)) and not p.startswith('.')]

include_dirs = [np.get_include()]
# define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
define_macros = []

# Cython compile all the things
ext_modules = [
    Extension('deep_hipsc_tracking.utils._poly',
              sources=['deep_hipsc_tracking/utils/_poly.pyx'],
              include_dirs=include_dirs,
              define_macros=define_macros),
    Extension("deep_hipsc_tracking.model._preproc",
              sources=["deep_hipsc_tracking/model/_preproc.pyx"],
              include_dirs=include_dirs,
              define_macros=define_macros),
    Extension("deep_hipsc_tracking.tracking._tracking",
              sources=["deep_hipsc_tracking/tracking/_tracking.pyx"],
              include_dirs=include_dirs,
              define_macros=define_macros),
    Extension('deep_hipsc_tracking.tracking._utils',
              sources=['deep_hipsc_tracking/tracking/_utils.pyx'],
              include_dirs=include_dirs,
              define_macros=define_macros),
    Extension("deep_hipsc_tracking.tracking._soft_assign",
              sources=["deep_hipsc_tracking/tracking/_soft_assign.pyx"],
              include_dirs=include_dirs,
              define_macros=define_macros),
    Extension('deep_hipsc_tracking.stats._grid_db',
              sources=['deep_hipsc_tracking/stats/_grid_db.pyx'],
              include_dirs=include_dirs,
              define_macros=define_macros),
]

setup(
    name=about['__package_name__'],
    version=about['__version__'],
    url=about['__url__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    ext_modules=cythonize(ext_modules, language_level=3),
    packages=('deep_hipsc_tracking',
              'deep_hipsc_tracking.model',
              'deep_hipsc_tracking.data',
              'deep_hipsc_tracking.plotting',
              'deep_hipsc_tracking.stats',
              'deep_hipsc_tracking.tracking',
              'deep_hipsc_tracking.utils'),
    scripts=scripts,
)

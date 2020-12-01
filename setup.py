#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup

NAME = 'rnn'
VERSION = '0.0.1'
AUTHOR = 'Sid Chaubal'
AUTHOR_EMAIL = 's.p.chaubal@student.vu.nl'
DESCRIPTION = 'A neural network, that will make you smile'
KEYWORDS = 'neural networks ai artificial intelligence'
PYTHON_VERSION = '>=3.7.0'
REQUIRED = ['pytorch']
EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = '\n' + f.read()

setup(
  name = NAME,
  version = VERSION,
  description = DESCRIPTION,
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = AUTHOR,
  author_email = AUTHOR_EMAIL,
  python_requires = PYTHON_VERSION,
  packages = find_packages(exclude = []),
  py_modules = ['architecture', 'scripts', 'architecture.data'],
  entry_points = {
  },
  install_requires = REQUIRED,
  extras_require = EXTRAS,
  include_package_data = True,
  license = 'MIT',
  classifiers = [
    'License :: OSI Approved :: MIT License',
    'Development Status :: 4 - Beta',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Artificial Intelligence'
  ]
)
#!/usr/bin/env python

import os


VERSION = '1.0.0'
NAME = 'osaic'
MODULES = [NAME]
DESCRIPTION = 'Create mosaics from images with ``python -mosaic image``'

URL = 'http://matteolandi.blogspot.com'

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.6',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

AUTHOR = 'Matteo Landi'
AUTHOR_EMAIL = 'landimatte@gmail.com'
KEYWORDS = "photo image mosaic creator".split(' ')

params = dict(
    name=NAME,
    version=VERSION,
    py_modules=MODULES,

    # metadata for upload to PyPI
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
)

from distutils.core import setup
setup(**params)

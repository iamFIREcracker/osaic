#!/usr/bin/env python

import os
from setuptools import setup


VERSION = '1.1.1'
NAME = 'osaic'
MODULES = [NAME]
DESCRIPTION = 'Create mosaics from images with ``python -mosaic image``'
readme = os.path.join(os.path.dirname(__file__), 'README.rst')
LONG_DESCRIPTION = open(readme).read()
requirements = os.path.join(os.path.dirname(__file__), 'requirements.txt')
INSTALL_REQUIRES = open(requirements).read().split()

URL = 'https://bitbucket.org/iamFIREcracker/osaic'

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.5',
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
    long_description=LONG_DESCRIPTION,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
    requires=INSTALL_REQUIRES,
    install_requires=INSTALL_REQUIRES,
)

setup(**params)

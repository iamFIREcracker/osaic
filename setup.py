#!/usr/bin/env python

import os
from setuptools import find_packages
from setuptools import setup


VERSION = '2.0.0'
NAME = 'osaic'
DESCRIPTION = 'Create mosaics from images with ``python -mosaic image``'
README = os.path.join(os.path.dirname(__file__), 'README.rst')
LONG_DESCRIPTION = open(README).read()
REQUIREMENTS = os.path.join(os.path.dirname(__file__), 'requirements.txt')
INSTALL_REQUIRES = open(REQUIREMENTS).read().split()

URL = 'https://bitbucket.org/iamFIREcracker/osaic'
DOWNLOAD_URL = 'http://pypi.python.org/pypi/osaic'

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

PARAMS = dict(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,

    # metadata for upload to PyPI
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license='BSD',
    keywords=KEYWORDS,
    url=URL,
    download_url=DOWNLOAD_URL,
    classifiers=CLASSIFIERS,
)

setup(**PARAMS)

=====
osaic
=====

**osaic** is a simple python module which let you create mosaics from
images by issuing a simple ``python -mosaic foo.jpg`` command.

The basic idea is to take as input a set of images: the first is used as
background for the final mosaic; the others are indexed by average
colors and pated, as tiles, over the final mosaic.

In addition, users are enabled to specify the final size of the mosaic
(relative to the size of the initial target image) and the number of
tiles to use per side.


Dependencies
============

The module depends on python ``PIL`` library (automatically fetched by
the installation script) for all the operations concerning *image
processing*. In addition, to add support for *jpeg*\s and *png*\s
images, and to enable the module to display mosaics on screen, please
install ``libjpeg``, ``libpng`` and ``libtk`` as well.  


Install
=======

To install **osaic**, you are enabled to grab it from both the mercurial
repository and from the Python Package Index (PyPI). The former is
preferred for *bleeding-edge* users, even tough the latter is not
guaranteed to be *very* stable as well.


Mercurial
---------

From sources::

    cd /wherever/you/want
    hg clone https://bitbucket.org/iamFIREcracker/osaic
    python setup.py install


PyPI
----

From sources::

    cd /path/to/workspace
    wget http://pypi.python.org/packages/source/o/osaic/osaic-2.0.0.tar.gz
    tar zxvf osaic-2.0.0.tar.gz
    cd osaic-2.0.0
    python setup.py install

From the PyPI::

    pip install osaic


Usage
=====

**osaic** is a module that can be used both as a standalone application
and as a standard python module.


Standalone application
----------------------

A typical usage of the application is to display a mosaic composition
created from a source image::

    python -mosaic image.jpg

If you want to save the output to a file instead of showing it::

    python -mosaic image.jpg -o mosaic-image.jpg

Finally, if you want to create a mosaic which is 4 times bigger than the
original image, and with 100 tiles per side, just issue::

    python -mosaic -z4 -t100 image.jpg

For everything else use the help message::

    python -mosaic -h


Library
-------

The module is a collection of objects and functions with different
capabilities: functions for vectors, color transformations, *image*
objects, image indexes and procedure wrappers.

Regarding operation with vectors, the module implement some basic
functions not included in standard Python but useful while working with
colors::

    >>> dotproduct([1, 2, 3], [4, 5, 6])
    70
    >>> difference([1, 2, 3], [1, 2, 3])
    [0, 0, 0]
    >>> squaredistance([1, 2, 3], [0, 0, 0])
    30

It is possible to find also a couple of functions wrapping up common
colors operations, like computation of the average color of an image and
color quantization. The latter is particularly useful while trying to
keep the CPU work load at low levels::

    >>> average_color('almost-red.png')
    (240, 10, 20)
    >>> quantize_color((240, 10, 20), levels=2)
    (192, 64, 64)

As noted earlier, the module is built on top of the Python ``PIL``
library. However, we chose not export such *external* objects, but
rather present to users some wrappers, namely `ImageWrapper`::

    >>> img = Image.open('foo.png')
    >>> img.size
    (640, 480)
    >>> img.reratio(5 / 1)
    >>> img.crop((0, 0, 10, 10))
    >>> img.size
    (10, 10)
    >>> img.show()

While creating mosaics, it comes in handy to have to possibility to
index a set of images and make it possible to search which of them is
the most similar, in terms of *average color*, to another one. The
``ImageList`` object is shipped with the module for this reason::

    >>> img_list = ImageList(['1.png', '2.png', '3.png'])
    >>> img_list.search((255, 0, 0))
    ImageTuple(filename='1.png', color=(255, 0, 0), image=None)

Finally, the module is shipped with a ``mosaicify`` function which wraps
up all the operations needed to create mosaic, including source images
indexing, and search of neighbour images depending on the average
color::

    >>> import osaic
    >>> osaic.mosaificy(
    ...     target='foo.png',
    ...     sources=['bar.png', 'asd.png', 'bazinga.png'],
    ...     tiles=128,
    ...     zoom=4,
    ...     output='mosaic.png',
    ... )

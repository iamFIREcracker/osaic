=====
osaic
=====
osaic is a simple module which let you create mosaics from images with
a simple `python -mosaic foo.jpg` command.


Dependencies
============
osaic depends on the ``PIL`` library which is going to be installed
automatically by the installation script: anyway, in order to make it
possible to edit ``jpeg`` and ``png`` images, please install ``libjpeg``
and ``libpng``.


Install
=======
From sources::

    cd /path/to/workspace
    wget http://pypi.python.org/packages/source/o/osaic/osaic-1.0.0.tar.gz
    tar zxvf osaic-1.0.0.tar.gz
    cd osaic-1.0.0
    python setup.py install

From the PyPI::

    pip install osaic


Usage
=====
osaic is a module that can be used either as a standalone application or
as a standard python library.

Standalone application
----------------------
A typical usage of the application is to display a mosaic composition
created from a source image::

    python -mosaic image.jpg

If you want to save the output to a file instead of showing it::

    python -mosaic image.jpg -o mosaic-image.jpg

Finally, if you want to create a mosaic which is 4 times bigger than the
original image, and with 100 tiles per axis, just issue::

    python -mosaic -s4 -t100 image.jpg

For everything else use the help message::

    python -mosaic -h


Library
-------
First of all, import the module::

    >>> import osaic

Then create a new ``Osaic`` object::

    >>> mos = osaic.Osaic('foo.jpg', tiles=32, size=1, mode=osaic.DEFAULT)

At this point, create the mosaic, show it on screen and save it on
a file::

    >>> mos.create()
    >>> mos.show()
    >>> mos.save('bar.jpg')

Alternatively, you can use the ``create`` function shipped with the
module which is a wrapper of all the actions listed above::

    >>> import osaic
    >>> osaic.create(
    ...     filename='foo.jpg',
    ...     tiles=32,
    ...     size=2,
    ...     mode=osaic.DEFAULT,
    ...     output=None,
    ... )

#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Create mosaics of input images.

The module offers the possibility to create poster-alike images compound
by small tiles representing input photos. Moreover, the module could be
used both as a standalone application and as a standard python library.

Given a list of input images, the first of them will be chosed as
target, i.e. the final image. On the other hand, other images will be
chosen in turn, modified, and finally placed in the right position of
the final mosaic image.

During the creation of a mosaic, we need to arrange the images used
a tiles, inside a data structure which make it possible to extract
images by color. Image, at least for early implementations, this
structure to be a simple list of images.

Moreover, in order to avoid long waits due to indexing of very large
images, we could implement a sort of filter, or a chain of filters, that
could eventually be used to scale down input images, or either quantize
their colors.

The next step is to analyze the target image and look for needed tiles
for the final mosaic. Depending on the specified number of
tiles-per-side, we are going to divide the original image in small
tiles. Then, for each tile, we are going to compute its *fingerprint*,
which in our case corresponds to its average color.

At this point everything is ready to actually create the mosaic. For
each tile extracted from the target image, look inside the efficient
data stucture and spot which of the available tiles has an average color
the most similar to the current one. Then we have to paste such found
tile in place of the original one.

Once we are done with all the tiles of the target image, we will be able
to either show the image on screen - watchout from large images ;-) - or
save it to a disk.

"""

from __future__ import division
import operator
from collections import namedtuple
from itertools import izip
from optparse import OptionParser
from optparse import OptionGroup
from random import randint

import Image
import ImageChops


"""Mode of quantization of color components."""
QUANTIZATION_MODES = 'bottom middle top'.split()


def dotproduct(vec1, vec2):
    """Return doct product of given vectors."""
    return sum(map(operator.mul, vec1, vec2))


def difference(vec1, vec2):
    """Return difference between given vectors."""
    return map(operator.sub, vec1, vec2)


def squaredistance(vec1, vec2):
    """Return the square distance between given vectors."""
    return sum(v ** 2 for v in difference(vec1, vec2))


def distance(vec1, vec2):
    """Return the distance between given vectors."""
    return squaredistance(vec1, vec2) ** 0.5


def average_color(img):
    """Return the average color of the given image.
    
    The calculus of the average color has been implemented by looking at
    each pixel of the image and accumulate each rgb component inside
    separate counters.
    
    """
    (width, height) = img.size
    (n, r, g, b) = (0, 0, 0, 0)
    for (many, color) in img:
        n += many
        r += many * color[0]
        g += many * color[1]
        b += many * color[2]
    return (r // n, g // n, b // n)


def quantize_color(color, levels, mode='middle'):
    """Reduce the spectrum of the given color.

    Each color component is forced to assume only certain values instead
    depending on the specified number of levels needed. If for example
    instead of 256 different levels, we need only a couple of them, each
    color component will be mapped into two ranges, namely [0, 128[ and
    [128, 256[.

    This way, given that multiple colors are possibly mapped on the same
    range of values and it is possible to decide to use as final output,
    the bottom, the middle or the top value of those ranges. Carrying on
    the example above, by default a color like (10, 10, 10), will match
    the first range of each component; hence, depending on the chosen
    mode, it will return (0, 0, 0), (64, 64, 64) or (127, 127, 127).


    """
    if levels <= 0 or levels > 256:
        raise ValueError("Number of levels should be in range ]0, 256].")
    if mode not in QUANTIZATION_MODES:
        raise ValueError("Mode should be one of %s." %
                            (' '.join(QUANTIZATION_MODES)))

    if levels == 256:
        return color

    if mode == 'top':
        inc = 256 // levels - 1
    elif mode == 'middle':
        inc = 256 // levels // 2
    else: # 'bottom'
        inc = 0

    # XXX refactor
    ret = [v * (levels - 1) // 255 * 256 // levels + inc for v in color]
    return tuple(ret)


def random_element(seq):
    """Return a random element of given sequence."""
    return seq[randint(0, len(seq) - 1)]


"""Object passed between different functions."""
ImageTuple = namedtuple('ImageTuple', 'filename color image'.split())


class ImageWrapper(object):
    """Wrapper around the ``Image`` object from the PIL library.

    We need to create our own image api and abstract, to the whole
    module layer, the inderlaying image processing library.

    """

    def __init__(self, **kwargs):
        """Initialize a new image object.

        It is possible both to open a new image from scratch, i.e. using
        its filename, or import raw data from another in-memory object.
        If both the ``filename`` and ``blob`` fields are specified, then
        the in-memory data associated to the image, will be taken from
        the blob.

        """
        self.filename = kwargs.pop('filename')
        self._blob = kwargs.pop('blob', None)
        if self.blob is None:
            try:
                self._blob = Image.open(self.filename)
            except IOError:
                raise

    def __iter__(self):
        """Iterate over the colors of the image.

        Return consecutive tuples containing the occurencies of a given
        color, a la ``groupby``, even tough the latter would return
        (color, n), instead of (n, color) like we do.

        TODO For very large images, we could think of using the color
        histogram. Consequently we need only to return an array
        containing only 256*3 values instead of the acqual width*height
        long.

        """
        (width, height) = self.size
        return iter(self._blob.getcolors(width * height))

    @property
    def blob(self):
        """Get image object as implemented by image library."""
        return self._blob

    @property
    def size(self):
        """Return a tuple representing the size of the image."""
        return self._blob.size

    def resize(self, size):
        """Set the size of the image."""
        if any(v < 0 for v in size):
            raise ValueError("Size could not contain negative values.")

        self._blob = self._blob.resize(size)

    @property
    def ratio(self):
        """Get the ratio (width / height) of the image."""
        (width, height) = self._blob.size
        return (width / height)

    def reratio(self, ratio):
        """Set the ratio (width / height) of the image.

        A consequence of the ratio modification, is image shrink; the
        size of the result image need to be modified to match desired
        ratio; consequently, part of the image will be thrown away.

        TODO while cropping an image to modify the ratio, we could
        probably use the imae baricenter in order to throw away useless
        parts of the image.

        """
        if ratio < 0:
            raise ValueError("Ratio could not assume negative values.")

        (width, height) = self.size
        if (width / height) > ratio:
            (new_width, new_height) = (int(ratio * height), height)
        else:
            (new_width, new_height) = (width, int(width / ratio))
        (x, y) = ((width - new_width) / 2, (height - new_height) / 2)
        rect = (x, y, x + new_width, y + new_height)
        self._blob = self._blob.crop(map(int, rect))

    def crop(self, rect):
        """Crop the image matching the given rectangle.

        The rectangle is a tuple containing top-left and bottom-right
        points: (x1, y1, x2, y2)

        """
        if any(v < 0 for v in rect):
            raise ValueError("Rectangle could not contain negative values.")
        return ImageWrapper(filename=self.filename, blob=self.blob.crop(rect))

    def show(self):
        """Display the image on screen."""
        self.blob.show()

    def save(self, filename):
        """Save the image onto the specified file."""
        self.blob.save(filename)


class ImageList(object):
    """List of images, optimized for color similarity searches.
    
    The class should be though as the implementation of a database of
    images; in particular, its implementation will be optimized for
    queries asking for similar images, where the similarity metric is
    based on the average color.

    TODO the current - dumb - implementation, is ``list`` based, but will
    be replaced with with a ``KD-Tree`` as soon as possible.

    """

    def __init__(self, iterable=None, **kwargs):
        """Initialize the internal list of images.

        Other than the list of filenames representing the images to
        index, it will come in handy to either preprocess or postprocess
        indexed images: hence users could specify ``prefunc`` and
        ``postfunc`` functions while creating a new list of images. In
        particular, in order to implement the possibility to pass
        additional arguments to filter functions, everything, included
        the functions, should be passed as *keyword* arguments.

        To group together similar images, we can quantize the average
        color of sources; in particular it is possible to specify the
        number of levels, ``qlevels``, each color component will have at
        the end of the whole process.

        """
        self._cache = dict()
        self.img_list = dict()
        prefunc = kwargs.pop('prefunc', None)
        postfunc = kwargs.pop('postfunc', None)
        qlevels = kwargs.pop('qlevels', 256)

        if iterable is None:
            raise ValueError("Empty image list.")

        for name in iterable:
            img = ImageWrapper(filename=name)

            if prefunc is not None:
                img = prefunc(img, **kwargs)

            color = quantize_color(average_color(img), qlevels)

            if postfunc is not None:
                img = postfunc(img, **kwargs)

            self.insert(ImageTuple(name, color, img))

    def __len__(self):
        """Get the length of the list of images."""
        return len(self.img_list)

    def insert(self, image):
        """Insert a new image in the list.
        
        Objects enqueued in the list are dictionaries containing the
        minimal amount of metadata required to handle images, namely the
        name of the image, its average color (we cache the value), and
        a blob object representing the raw processed image. Note that
        after the application of the ``postfunc`` filter, it is possible
        for the blob object to be None.

        """
        self.img_list.setdefault(image.color, list()).append(image)

    def search(self, color):
        """Search the most similar image in terms of average color."""
        if color in self._cache:
            return random_element(self._cache[color])

        best_item = None
        best_dist = None
        for (item_color, item) in self.img_list.iteritems():
            dist = squaredistance(color, item_color)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_item = item
        self._cache[color] = best_item
        return random_element(best_item)


def resizefunc(img, **kwargs):
    """Adjust the size of the given image.

    First, the ratio of the image is modified in order to match an
    eventually specified one. Then the size of the image is modified
    accordingly.

    """
    ratio = kwargs.pop('ratio', None)
    size = kwargs.pop('size', None)

    if ratio is not None:
        img.reratio(ratio)

    if size is not None:
        img.resize(size)

    return img


def voidfunc(img, **kwargs):
    """Do nothing special from returning the image as is."""
    return img


def deletefunc(img, **kwargs):
    """Delete input image and return None.
    
    GC will do all the magic for us, hence we have nothing special to do
    here: just return None, or *pass*.
    
    """
    pass



def tilefy(img, tiles):
    """Convert input image into a matrix of tiles.

    Return a matrix composed by tile-objects, i.e. dictionaries,
    containing useful information for the final mosaic.
    
    In our particolar case we are in need of the average color of the
    region representing a specific tile. For compatibility with the
    objects used for the ``ImageList`` we set the filename and blob
    fields eithter.

    """
    matrix = [[None for i in xrange(tiles)] for j in xrange(tiles)]
    (width, height) = img.size
    (tile_width, tile_height) = (width // tiles, height // tiles)
    (x, y) = (0, 0)
    for (i, y) in enumerate(xrange(0, tile_height * tiles, tile_height)):
        for (j, x) in enumerate(xrange(0, tile_width * tiles, tile_width)):
            rect = (x, y, x + tile_width, y + tile_height)
            tile = img.crop(rect)
            matrix[i][j] = ImageTuple(img.filename, average_color(tile), None)
    return matrix


def mosaicify(target, sources, tiles=32, zoom=1, levels=256, output=None):
    """Create mosaic of photos.
    
    The function wraps all process of the creation of a mosaic, given
    the target, the list of source images, the number of tiles to use
    per side, the zoom level (a.k.a.  how large the mosaic will be),
    whether we are interested or not in color quantization, and finally
    if we want to display the output on screen or dump it on a file.

    First, open the target image, divide it into the specified number of
    tiles, and store information about the tiles average color. In
    orther to reduce the amount of used memory, we will free the *blobs*
    associated to each processed image, as soon as possible, aka inside
    the ``postfunc`` function.

    Then, index all the source images by color. Given that we are aware
    about the size and the ratio of the tiles of the target, we can use
    the ``prefunc`` to reduce the dimension of the image; consequently
    the amount of computations needed to compute the avergage color will
    smaller. Moreover, as in the previous paragraph, there is no need to
    keep into processed images, hence we are going to use the
    ``postfunc`` method to delete them.

    Finally, for each tile extranted from the target image, we need to
    find the most similar contained inside the list of source images,
    and paste it in the right position inside the mosaic image.

    When done, show the result on screen or dump it on the disk.

    """
    # open target image, and divide it into tiles..
    img = ImageWrapper(filename=target)
    tile_matrix = tilefy(img, tiles)
    # ..process and sort all source tiles..
    tile_ratio = img.ratio
    (width, height) = img.size
    (tile_width, tile_height) = (zoom * width // tiles, zoom * height // tiles)
    tile_size = (tile_width, tile_height)
    source_list = ImageList(sources, prefunc=resizefunc, postfunc=voidfunc,
                            ratio=tile_ratio, size=tile_size, qlevels=levels)
    # ..prepare output image..
    (mosaic_width, mosaic_height) = (tiles * tile_width, tiles * tile_height)
    mosaic_size = (mosaic_width, mosaic_height)
    img.resize(mosaic_size)
    # ..and start to paste tiles
    for (i, tile_row) in enumerate(tile_matrix):
        for (j, tile) in enumerate(tile_row):
            (x, y) = (tile_width * j, tile_height * i)
            rect = (x, y, x + tile_width, y + tile_height)
            closest = source_list.search(quantize_color(tile.color, levels))
            closest_img = closest.image
            img._blob.paste(closest_img._blob, rect) # XXX hack
    # finally show the result, or dump it on a file.
    if output is None:
        img.show()
    else:
        img.save(output)



def _build_parser():
    """Return a command-line arguments parser."""
    usage = "Usage: %prog [-t TILES] [-z ZOOM] [-l LEVELS] [-o OUTPUT] IMAGE1 ..."
    parser = OptionParser(usage=usage)

    config = OptionGroup(parser, "Configuration Options")
    config.add_option("-t", "--tiles", dest="tiles", default="32",
                      help="Number of tiles per side.", metavar="TILES")
    config.add_option("-z", "--zoom", dest="zoom", default="1",
                      help="Zoom level of the mosaic.", metavar="ZOOM")
    config.add_option("-l", "--levels", dest="levels", default="256",
                      help="Color quantization levels, per component",
                      metavar="LEVELS")
    config.add_option("-o", "--output", dest="output", default=None,
                      help="Save output instead of showing it.",
                      metavar="OUTPUT")
    parser.add_option_group(config)

    return parser


def _main():
    """Run the command-line interface."""
    parser = _build_parser()
    (options, args) = parser.parse_args()

    if not args:
        parser.print_help()
        exit(1)

    mosaicify(
        target=args[0],
        sources=set(args[1:] or args),
        tiles=int(options.tiles),
        zoom=int(options.zoom),
        levels=int(options.levels),
        output=options.output,
    )


if __name__ == '__main__':
    _main()

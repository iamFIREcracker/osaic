#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Create mosaics of input images.
"""

from __future__ import division
from optparse import OptionParser
from optparse import OptionGroup
from random import randint

import Image
import ImageChops



def random_element(seq):
    """Return a random element from the given sequence."""
    return seq[randint(0, len(seq) - 1)]


def tileify(image, tiles, zoom):
    """Transform the given image into a collection of tiles.
    
    Accepted keywords:
        image image to tileify.
        tiles how many tiles to use per dimension.
        zoom zoom factor handy for resizing the mosaic.
    """
    (width, height) = image.size
    image = image.resize((width * zoom, height * zoom))

    (tile_w, tile_h) = (width * zoom // tiles, height * zoom // tiles)

    mosaic = [[None for i in xrange(tiles)] for j in xrange(tiles)]
    for i in xrange(tiles):
        for j in xrange(tiles):
            (x, y) = (j * tile_w, i * tile_h)
            mosaic[i][j] = Tile(image.crop((x, y, x + tile_w, y + tile_h)))
    return mosaic
    

def untileify(mosaic):
    """Transform the given mosaic into an image.

    Accepted keywords:
        mosaic collection of tiles created using ``tileify``.
    """
    (tile_w, tile_h) = mosaic[0][0].size
    tiles = len(mosaic)

    image = Image.new("RGB", (tile_w * tiles, tile_h * tiles))
    for i in xrange(len(mosaic)):
        for j in xrange(len(mosaic[0])):
            # pate the tile to the output surface
            (x, y) = (j * tile_w, i * tile_h)
            image.paste(mosaic[i][j].image, (x, y, x + tile_w, y + tile_h))
    return image


def mosaicify(target, sources, tiles=32, zoom=1, output=None):
    """XXX"""
    mosaic = Mosaic(target, tiles, zoom)

    source_tiles = list()
    for source in sources:
        try:
            tile = Tile(Image.open(source))
            tile.ratio = mosaic.tileratio
            tile.size = mosaic.tilesize
            source_tiles.append(tile)
        except IOError:
            # Let's try to go on without the failing image: just
            # print a message.
            print "Unable to open: %s" % (source)
    if not source_tiles:
        raise ValueError("The tile list cannot be empty.")

    for tile in mosaic:
        color = tile.average_color
        new_tile = random_element(source_tiles).colorify(color)
        tile.paste(new_tile)
        
    if output:
        mosaic.save(output)
    else:
        mosaic.show()



class InvalidInput(Exception):
    """Raised when the input image can not be read.
    """
    def __init__(self, input_):
        super(InvalidInput, self).__init__()
        self.input_ = input_


class InvalidOutput(Exception):
    """Raised when the output image can not be written.
    """
    def __init__(self, output):
        super(InvalidOutput, self).__init__()
        self.output = output


class Tile(object):
    """XXX"""
    
    def __init__(self, image):
        """Initialize the underlaying image object.
        
        Accepted keywords:
            image image object to use as tile.
        """
        self.image = image

    @property
    def average_color(self):
        """Return the average color of the tile."""
        (width, height) = self.image.size
        (N, R, G, B) = (0, 0, 0, 0)
        for (n, (r, g, b)) in self.image.getcolors(width * height):
            N += n
            R += n * r
            G += n * g
            B += n * b
        return (R // N, G // N, B // N)

    @property
    def ratio(self):
        """Return the ratio (width / height) of tile."""
        (width, height) = self.size
        return (width / height)

    @ratio.setter
    def ratio(self, ratio):
        """Set the ratio (width / height) of the tile."""
        (width, height) = self.size
        if (width / height) > ratio:
            (width, height) = (ratio * height, height)
        else:
            (width, height) = (width, width / ratio)
        self.image = self.image.crop((0, 0, int(width), int(height)))

    @property
    def size(self):
        """Return a tuple representing the size of the tile."""
        return self.image.size

    @size.setter
    def size(self, size):
        """Set the size of the tile."""
        self.image = self.image.resize(size)

    def paste(self, tile):
        """Substitute the content of the tile with the given one.

        Accepted keywords:
            tile a Tile object.
        """
        (width, height) = tile.size
        self.image.paste(tile.image, (0, 0, width, height))

    def colorify(self, color):
        """Apply a colored layer over the tile.
        
        Accepted keywords:
            color tuple containing the RGB values of the layer.

        Return:
            The new colored Tile.
        """
        overlay = Image.new("RGB", self.size, color)
        return Tile(ImageChops.multiply(self.image, overlay))


class Mosaic(object):
    """XXX"""

    def __init__(self, target, tiles=32, zoom=1):
        """Initialize the rendering object.
        
        Accepted keywords:
            target name of the file we which to *mosaicify*.
            tiles number of tiles to use per dimention.
            zoom zoom factor handy for resizing the mosaic.

        Raise:
            InvalidInput, ValueError.
        """
        if tiles <= 0:
            raise ValueError("The number of tiles cannot be smaller than 0.")
        if zoom <= 0:
            raise ValueError("Zoom level cannot be smaller than 0.")

        try:
            self.tiles = tileify(Image.open(target), tiles, zoom)
        except IOError:
            raise InvalidInput(target)

        # internal state used while iterating over the tiles.
        self.i = self.j = 0

    def __iter__(self):
        """Add iteration support."""
        return self

    def next(self):
        """Return one by one the tiles used by the mosaic.

        The internal state is held by ``i`` and ``j`` variables.

        Return:
            The next tile used.

        Raise:
            StopIteration.
        """
        if self.i == len(self.tiles):
            self.i == self.j == 0
            raise StopIteration()

        tile = self.tiles[self.i][self.j]

        self.j += 1
        if self.j == len(self.tiles):
            self.i += 1
            self.j = 0

        return tile

    @property
    def tileratio(self):
        """Return the ratio (width / height) of the used tiles."""
        (width, height) = self.tilesize
        return (width / height)

    @property
    def tilesize(self):
        """Return the size of used tiles."""
        return self.tiles[0][0].size

    def save(self, filename):
        """Save the mosaic on a file.

        Accepted keywords:
            filename path of the destination file to create.

        Raise:
            InvalidOutput.
        """
        try:
            untileify(self.tiles).save(filename)
        except IOError:
            raise InvalidOutput(filename)

    def show(self):
        """show the mosaic on screen."""
        untileify(self.tiles).show()



def _build_parser():
    """Return a command-line arguments parser."""
    usage = "Usage: %prog [-t TILES] [-z ZOOM] [-o OUTPUT] IMAGE1 ..."
    parser = OptionParser(usage=usage)

    config = OptionGroup(parser, "Configuration Options")
    config.add_option("-t", "--tiles", dest="tiles", default="32",
                      help="Number of tiles per side.", metavar="TILES")
    config.add_option("-z", "--zoom", dest="zoom", default="1",
                      help="Zoom level of the mosaic.", metavar="ZOOM")
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

    try:
        mosaicify(
            target=args[0],
            sources=set(args[1:]),
            tiles=int(options.tiles),
            zoom=int(options.zoom),
            output=options.output,
        )
    except InvalidInput, e:
        print "Input image '%s' can not be read." % e.input_
    except InvalidOutput, e:
        print "Output image '%s' can not be written." % e.output


if __name__ == '__main__':
    _main()

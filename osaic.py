#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Create mosaics of input images.
"""

from __future__ import division
from optparse import OptionParser
from optparse import OptionGroup

import Image
import ImageChops


# Accepted mosaic modes.
MODES = 2
DEFAULT, EIGHTBIT = range(MODES)



def average_color(image):
    """Return the average color of the given image."""
    (width, height) = image.size
    #(N, R, G, B) = (0, 0, 0, 0)
    #for (n, (r, g, b)) in img.getcolors(img.size[0] * img.size[1]):
        #N += n * n
        #R += n * r
        #G += n * g
        #B += n * b
    #return (R // N, G // N, B // N)
    (N, (R, G, B)) = max(image.getcolors(width * height))
    return (R, G, B)


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


class Osaic(object):
    """Core object of the module.

    Given an input image and a couple of optional settings, create
    a mosaic and display on screen or write it to disk.
    """

    def __init__(self, filename, tiles=32, size=1, mode=DEFAULT):
        """Initialize the rendering object.

        Accepted keywords:
            filename: source input image.
            tiles: number of tiles to use per dimension (width, height).
            zoom: size of the mosaic (compared to original photo).
            mode: the kind of mosaic to create.

        Raise:
            InvalidInput, ValueError.
        """
        try:
            self.source = Image.open(filename)
        except IOError:
            raise InvalidInput(filename)

        if tiles <= 0:
            raise ValueError("'tiles' should be greater than 0.")
        self.tiles = tiles

        if size <= 0:
            raise ValueError("'size' should be greater than 0.")
        self.size = size

        if mode not in range(MODES):
            raise ValueError("'mode' should be in the range [0, %d[" % MODES)
        self.mode = mode

        self.mosaic = None

    def create(self):
        """Create the mosaic using the specified ``mode`` and finally
        store the output inside the ``mosaic`` variable.
        """
        # size of the tiles extracted from the source image.
        (src_tile_w, src_tile_h) = map(lambda v: v // self.tiles,
                                       self.source.size)
        # size of the tiles used for the mosaic: these values are
        # different from the ones of the source image because of the
        # ``size`` value that acts like a zoom effect.
        (mos_tile_w, mos_tile_h) = map(lambda v: v * self.size // self.tiles,
                                       self.source.size)

        # size of the mosaic.
        (mos_w, mos_h) = (mos_tile_w * self.tiles, mos_tile_h * self.tiles)
        mosaic = Image.new("RGBA", (mos_w, mos_h))

        for i in xrange(self.tiles):
            for j in xrange(self.tiles):
                (x, y) = (j * src_tile_w, i * src_tile_h)

                # get the tile average color ..
                cropped = self.source.crop((x, y, x + src_tile_w,
                                            y + src_tile_h))
                (r, g, b) = average_color(cropped)
                # .. and create a monochromatic surface.
                color = Image.new("RGB", (mos_tile_w, mos_tile_h), (r, g, b))

                # elaborate the new mosaic tile
                if self.mode == DEFAULT:
                    resized = self.source.resize((mos_tile_w, mos_tile_h))
                    tile = ImageChops.multiply(resized, color)
                elif self.mode == EIGHTBIT:
                    tile = color
                
                # pate the tile to the output surface
                (x, y) = (j * mos_tile_w, i * mos_tile_h)
                mosaic.paste(tile, (x, y, x + mos_tile_w, y + mos_tile_h))

        self.mosaic = mosaic

    def save(self, filename):
        """Save the mosaic on a file.

        Accepted keywords:
            filename: path of the destination file to create.

        Raise:
            InvalidOutput.
        """
        try:
            self.mosaic.save(filename)
        except (IOError, KeyError):
            raise InvalidOutput(filename)

    def show(self):
        """Show the mosaic on screen."""
        self.mosaic.show()


def create(filename, tiles, size, mode, output):
    """Wrapper of the ``Osaic`` object."""
    try:
        osaic = Osaic(filename, tiles, size, mode)
        osaic.create()
        if output:
            osaic.save(output)
        else:
            osaic.show()
    except InvalidInput, e:
        print "Input image '%s' can not be read." % e.input_
    except InvalidOutput, e:
        print "Output image '%s' can not be written." % e.output



def _build_parser():
    """Return a command-line arguments parser."""
    usage = "Usage: %prog [-t TILES] [-s SIZE] [-m MODE] [-o OUTPUT] IMAGE"
    parser = OptionParser(usage=usage)

    config = OptionGroup(parser, "Configuration Options")
    config.add_option("-t", "--tiles", dest="tiles", default="32",
                      help="Number of tiles per width/height", metavar="TILES")
    config.add_option("-s", "--size", dest="size", default="1",
                      help="Size of the mosaic.", metavar="SIZE")
    config.add_option("-m", "--mode", dest="mode", default="0",
                      help="Type of mosaic to create.", metavar="MODE")
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

    create(
        filename=args[0],
        tiles=int(options.tiles),
        size=int(options.size),
        mode=int(options.mode),
        output=options.output,
    )


if __name__ == '__main__':
    _main()

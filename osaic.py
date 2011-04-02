#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""Create mosaics of input images.
"""

from __future__ import division
from optparse import OptionParser
from optparse import OptionGroup

import Image
import ImageChops



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


class Mosaic(object):
    """Core object of the module.

    Given an input image and a couple of optional settings, create
    a mosaic and display on screen or write it to disk.
    """

    def __init__(self, filenames, tiles=32, size=1):
        """Initialize the rendering object.

        Accepted keywords:
            filenames: list of filenames of the photos to use for the
                       mosaic.
            tiles: number of tiles to use per dimension (width, height).
            zoom: size of the mosaic (compared to original photo).

        Raise:
            InvalidInput, ValueError.
        """
        if not filenames:
            raise ValueError("'filenames' should be not empty.");
        self.sources = list()
        for fname in filenames:
            try:
                self.sources.append(Image.open(fname))
            except IOError:
                raise InvalidInput(fname)

        if tiles <= 0:
            raise ValueError("'tiles' should be greater than 0.")
        self.tiles = tiles

        if size <= 0:
            raise ValueError("'size' should be greater than 0.")
        self.size = size

        self.mosaic = None

    def create(self):
        """Create the mosaic and store the output inside the ``mosaic``
        variable.
        """
        # size of the tiles extracted from the source image.
        (src_tile_w, src_tile_h) = map(lambda v: v // self.tiles,
                                       self.sources[0].size)
        # size of the tiles used for the mosaic: these values are
        # different from the ones of the source image because of the
        # ``size`` value that acts like a zoom effect.
        (mos_tile_w, mos_tile_h) = map(lambda v: v * self.size // self.tiles,
                                       self.sources[0].size)

        # size of the mosaic.
        (mos_w, mos_h) = (mos_tile_w * self.tiles, mos_tile_h * self.tiles)
        mosaic = Image.new("RGBA", (mos_w, mos_h))

        for i in xrange(self.tiles):
            for j in xrange(self.tiles):
                (x, y) = (j * src_tile_w, i * src_tile_h)

                # get the tile average color ..
                cropped = self.sources[0].crop((x, y, x + src_tile_w,
                                               y + src_tile_h))
                (r, g, b) = average_color(cropped)
                # .. and create a monochromatic surface.
                color = Image.new("RGB", (mos_tile_w, mos_tile_h), (r, g, b))

                # elaborate the new mosaic tile
                resized = self.sources[0].resize((mos_tile_w, mos_tile_h))
                tile = ImageChops.multiply(resized, color)

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


def create(filenames, tiles=32, size=1, output=None):
    """Wrapper of the ``Mosaic`` object."""
    try:
        mosaic = Mosaic(filenames, tiles, size)
        mosaic.create()
        if output:
            mosaic.save(output)
        else:
            mosaic.show()
    except InvalidInput, e:
        print "Input image '%s' can not be read." % e.input_
    except InvalidOutput, e:
        print "Output image '%s' can not be written." % e.output



def _build_parser():
    """Return a command-line arguments parser."""
    usage = "Usage: %prog [-t TILES] [-s SIZE] [-o OUTPUT] IMAGE1 ..."
    parser = OptionParser(usage=usage)

    config = OptionGroup(parser, "Configuration Options")
    config.add_option("-t", "--tiles", dest="tiles", default="32",
                      help="Number of tiles per width/height.", metavar="TILES")
    config.add_option("-s", "--size", dest="size", default="1",
                      help="Size of the mosaic.", metavar="SIZE")
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
        filenames=args,
        tiles=int(options.tiles),
        size=int(options.size),
        output=options.output,
    )


if __name__ == '__main__':
    _main()

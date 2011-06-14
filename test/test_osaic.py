#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import unittest

import Image

from osaic import ImageWrapper
from osaic import ImageList


COLORS = 'red green blue'
IMGCOLORS = None


def imgcolors_init():
    global IMGCOLORS
    img_list = []
    for color in COLORS.split():
        filename = os.path.join(sys.path[0], color + '.png')
        img = Image.new("RGB", (288, 288), color)
        img.save(filename)
        img_list.append(filename)
    IMGCOLORS = img_list


def imgcolors_fini():
    for filename in IMGCOLORS:
        os.unlink(filename)

def setUpModule():
    imgcolors_init()

def tearDownModule():
    imgcolors_fini()


class TestImageWrapper(unittest.TestCase):

    def test_init(self):
        # ``filename`` is mandatory.
        self.assertRaises(KeyError, ImageWrapper)
        # open a not existing image, hence raise IOError
        self.assertRaises(IOError, ImageWrapper, filename='foo')
        # open an existing file.
        img = ImageWrapper(filename=IMGCOLORS[0])
        # open a not existing image but specify the blob field;
        img1 = ImageWrapper(filename='foo', blob=img.blob)

    def test_size(self):
        img = ImageWrapper(filename=IMGCOLORS[0])
        # double initial size
        (width, height) = img.size
        (dwidth, dheight) = (2 * width, 2 * height)
        img.resize((dwidth, dheight))
        self.assertEqual((dwidth, dheight), img.size)
        # check wrong values
        img.resize((0, 0))
        self.assertRaises(ValueError, img.resize, (-1, 0))
        self.assertRaises(ValueError, img.resize, (0, -1))
        self.assertRaises(ValueError, img.resize, (-100, -100))

    def test_ratio(self):
        img = ImageWrapper(filename=IMGCOLORS[0])
        # double initial size, keeping ratio constant
        ratio = img.ratio
        (width, height) = img.size
        (dwidth, dheight) = (2 * width, 2 * height)
        img.resize((dwidth, dheight))
        self.assertEqual(ratio, img.ratio)
        # invert width and height dimensions
        img.resize((height, width))
        self.assertEqual(1 / ratio, img.ratio)
        # change the ratio
        nratio = 3 / 1
        img.reratio(nratio)
        self.assertEqual(nratio, img.ratio)
        # check wrong values
        self.assertRaises(ValueError, img.reratio, -1)
        self.assertRaises(ValueError, img.reratio, -1000)

    def test_crop(self):
        img = ImageWrapper(filename=IMGCOLORS[0])
        # crop half width
        (width, height) = img.size
        rect = (0, 0, width // 2, height)
        img1 = img.crop(rect)
        self.assertEqual((width // 2, height), img1.size)
        # crop half height
        (width, height) = img.size
        rect = (0, 0, width, height // 2)
        img1 = img.crop(rect)
        self.assertEqual((width, height // 2), img1.size)
        # finally crop both dimensions
        (width, height) = img.size
        rect = (0, 0, width // 2, height // 2)
        img1 = img.crop(rect)
        self.assertEqual((width // 2, height // 2), img1.size)

    def test_show_and_save(self):
        img = ImageWrapper(filename=IMGCOLORS[0])
        # it is not easy to test such a feature: for the moment just for
        # the method presence. XXX
        self.assertEquals(True, hasattr(img, 'show'))
        # same as before, but for the ``save`` method. Too lazy for
        # these ;-) XXX
        self.assertEquals(True, hasattr(img, 'save'))



class TestImageList(unittest.TestCase):

    def test_init(self):
        def count(img, **kwargs):
            kwargs['foo'][0] += 1
            return img
        # the list of source images is mandatory
        self.assertRaises(ValueError, ImageList)
        # create a list of images composed by a single element; in
        # addiction verify that both ``prefunc`` and ``postfunc`` get
        # invoked one time each.
        sources = IMGCOLORS[0:1]
        counter = [0] # we need to pass an integer by reference ;-)
        il = ImageList(sources, prefunc=count, postfunc=count, foo=counter)
        self.assertEqual(len(sources), len(il))
        self.assertEqual(2 * len(sources), counter[0])
        # now with the whole color array
        sources = IMGCOLORS
        counter = [0] # we need to pass an integer by reference ;-)
        ImageList(sources, prefunc=count, postfunc=count, foo=counter)
        self.assertEqual(2 * len(sources), counter[0])

    def test_search(self):
        def void(img, **kwargs):
            return img
        def skip(img, **kwargs):
            pass
        sources = IMGCOLORS
        il = ImageList(sources, prefunc=void, postfunc=skip)
        # search a red picture.
        img_tuple = il.search((255, 0, 0))
        self.assertEqual(IMGCOLORS[0], img_tuple.filename)
        self.assertEqual((255, 0, 0), img_tuple.color)
        self.assertEqual(None, img_tuple.image)
        # now a green one..
        img_tuple = il.search((0, 255, 0))
        self.assertEqual(IMGCOLORS[1], img_tuple.filename)
        # finally the blue one
        img_tuple = il.search((0, 0, 255))
        self.assertEqual(IMGCOLORS[2], img_tuple.filename)



if __name__ == '__main__':
    unittest.main()

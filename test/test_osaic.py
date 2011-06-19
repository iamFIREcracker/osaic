#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import unittest

import Image

from osaic import dotproduct
from osaic import difference
from osaic import squaredistance
from osaic import average_color
from osaic import quantize_color
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



class TestFunctions(unittest.TestCase):

    def test_vectors(self):
        v1 = [1, 2, 3, 4]
        v2 = [5, 6, 7, 8]
        v3 = [0, 0, 0, 0]
        # dot product
        self.assertEquals(70, dotproduct(v1, v2))
        self.assertEquals(0, dotproduct(v1, v3))
        # vector difference
        self.assertEquals([4, 4, 4, 4], difference(v2, v1))
        self.assertEquals(v2, difference(v2, v3))
        # squaredisance
        self.assertEquals(30, squaredistance(v1, v3))
        self.assertEquals(64, squaredistance(v2, v1))

    def test_average(self):
        # XXX
        pass

    def test_quantize(self):
        red = (255, 0, 0)
        # sanity checks
        self.assertRaises(ValueError, quantize_color, red, 0)
        self.assertRaises(ValueError, quantize_color, red, 257)
        self.assertRaises(ValueError, quantize_color, red, 128, 'asd')
        # noop
        self.assertEquals(red, quantize_color(red, 256))
        # misc quantization
        self.assertEquals((192, 0, 0), quantize_color(red, 4, 'bottom'))
        self.assertEquals((224, 32, 32), quantize_color(red, 4, 'middle'))
        self.assertEquals((255, 63, 63), quantize_color(red, 4, 'top'))
        self.assertEquals((192, 64, 64), quantize_color((240, 10, 20), 2, 'middle'))



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

    def test_paste(self):
        img = ImageWrapper(filename=IMGCOLORS[0])
        img1 = ImageWrapper(filename=IMGCOLORS[1])
        # paste ``img`` on the first on the left part of ``img1``
        color = average_color(img)
        color1 = average_color(img1)
        (width, height) = img1.size
        (new_width, new_height) = (width // 2, height)
        img.resize((new_width, new_height))
        img1.paste(img, (0, 0, new_width, new_height))
        self.assertNotEquals(color1, average_color(img1))
        # paste it again on the right.
        img1.paste(img, (new_width, 0, width, height))
        self.assertEquals(color, average_color(img1))

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

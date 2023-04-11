#
# The Python Imaging Library.
# $Id$
#
# sequence support classes
#
# history:
# 1997-02-20 fl     Created
#
# Copyright (c) 1997 by Secret Labs AB.
# Copyright (c) 1997 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

##


class Iterator:
    """
    This class implements an iterator object that can be used to loop
    over an image sequence.

    You can use the ``[]`` operator to access elements by index. This operator
    will raise an :py:exc:`IndexError` if you try to access a nonexistent
    frame.

    :param im: An image object.
    """

    def __init__(self, im):
        if not hasattr(im, "seek"):
            msg = "im must have seek method"
            raise AttributeError(msg)
        self.im = im
        self.position = getattr(self.im, "_min_frame", 0)

    def __getitem__(self, ix):
        try:
            self.im.seek(ix)
            return self.im
        except EOFError as e:
            raise IndexError from e  # end of sequence

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.im.seek(self.position)
            self.position += 1
            return self.im
        except EOFError as e:
            raise StopIteration from e


def all_frames(im, func=None):
    """
    Applies a given function to all frames in an image or a list of images.
    The frames are returned as a list of separate images.

    :param im: An image, or a list of images.
    :param func: The function to apply to all of the image frames.
    :returns: A list of images.
    """
    if not isinstance(im, list):
        im = [im]

    ims = []
    for imSequence in im:
        current = imSequence.tell()

        ims += [im_frame.copy() for im_frame in Iterator(imSequence)]

        imSequence.seek(current)
    return [func(im) for im in ims] if func else ims

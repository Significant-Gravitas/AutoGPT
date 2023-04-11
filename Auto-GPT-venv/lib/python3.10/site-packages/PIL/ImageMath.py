#
# The Python Imaging Library
# $Id$
#
# a simple math add-on for the Python Imaging Library
#
# History:
# 1999-02-15 fl   Original PIL Plus release
# 2005-05-05 fl   Simplified and cleaned up for PIL 1.1.6
# 2005-09-12 fl   Fixed int() and float() for Python 2.4.1
#
# Copyright (c) 1999-2005 by Secret Labs AB
# Copyright (c) 2005 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import builtins

from . import Image, _imagingmath


def _isconstant(v):
    return isinstance(v, (int, float))


class _Operand:
    """Wraps an image operand, providing standard operators"""

    def __init__(self, im):
        self.im = im

    def __fixup(self, im1):
        # convert image to suitable mode
        if isinstance(im1, _Operand):
            # argument was an image.
            if im1.im.mode in ("1", "L"):
                return im1.im.convert("I")
            elif im1.im.mode in ("I", "F"):
                return im1.im
            else:
                msg = f"unsupported mode: {im1.im.mode}"
                raise ValueError(msg)
        else:
            # argument was a constant
            if _isconstant(im1) and self.im.mode in ("1", "L", "I"):
                return Image.new("I", self.im.size, im1)
            else:
                return Image.new("F", self.im.size, im1)

    def apply(self, op, im1, im2=None, mode=None):
        im1 = self.__fixup(im1)
        if im2 is None:
            # unary operation
            out = Image.new(mode or im1.mode, im1.size, None)
            im1.load()
            try:
                op = getattr(_imagingmath, op + "_" + im1.mode)
            except AttributeError as e:
                msg = f"bad operand type for '{op}'"
                raise TypeError(msg) from e
            _imagingmath.unop(op, out.im.id, im1.im.id)
        else:
            # binary operation
            im2 = self.__fixup(im2)
            if im1.mode != im2.mode:
                # convert both arguments to floating point
                if im1.mode != "F":
                    im1 = im1.convert("F")
                if im2.mode != "F":
                    im2 = im2.convert("F")
            if im1.size != im2.size:
                # crop both arguments to a common size
                size = (min(im1.size[0], im2.size[0]), min(im1.size[1], im2.size[1]))
                if im1.size != size:
                    im1 = im1.crop((0, 0) + size)
                if im2.size != size:
                    im2 = im2.crop((0, 0) + size)
            out = Image.new(mode or im1.mode, im1.size, None)
            im1.load()
            im2.load()
            try:
                op = getattr(_imagingmath, op + "_" + im1.mode)
            except AttributeError as e:
                msg = f"bad operand type for '{op}'"
                raise TypeError(msg) from e
            _imagingmath.binop(op, out.im.id, im1.im.id, im2.im.id)
        return _Operand(out)

    # unary operators
    def __bool__(self):
        # an image is "true" if it contains at least one non-zero pixel
        return self.im.getbbox() is not None

    def __abs__(self):
        return self.apply("abs", self)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.apply("neg", self)

    # binary operators
    def __add__(self, other):
        return self.apply("add", self, other)

    def __radd__(self, other):
        return self.apply("add", other, self)

    def __sub__(self, other):
        return self.apply("sub", self, other)

    def __rsub__(self, other):
        return self.apply("sub", other, self)

    def __mul__(self, other):
        return self.apply("mul", self, other)

    def __rmul__(self, other):
        return self.apply("mul", other, self)

    def __truediv__(self, other):
        return self.apply("div", self, other)

    def __rtruediv__(self, other):
        return self.apply("div", other, self)

    def __mod__(self, other):
        return self.apply("mod", self, other)

    def __rmod__(self, other):
        return self.apply("mod", other, self)

    def __pow__(self, other):
        return self.apply("pow", self, other)

    def __rpow__(self, other):
        return self.apply("pow", other, self)

    # bitwise
    def __invert__(self):
        return self.apply("invert", self)

    def __and__(self, other):
        return self.apply("and", self, other)

    def __rand__(self, other):
        return self.apply("and", other, self)

    def __or__(self, other):
        return self.apply("or", self, other)

    def __ror__(self, other):
        return self.apply("or", other, self)

    def __xor__(self, other):
        return self.apply("xor", self, other)

    def __rxor__(self, other):
        return self.apply("xor", other, self)

    def __lshift__(self, other):
        return self.apply("lshift", self, other)

    def __rshift__(self, other):
        return self.apply("rshift", self, other)

    # logical
    def __eq__(self, other):
        return self.apply("eq", self, other)

    def __ne__(self, other):
        return self.apply("ne", self, other)

    def __lt__(self, other):
        return self.apply("lt", self, other)

    def __le__(self, other):
        return self.apply("le", self, other)

    def __gt__(self, other):
        return self.apply("gt", self, other)

    def __ge__(self, other):
        return self.apply("ge", self, other)


# conversions
def imagemath_int(self):
    return _Operand(self.im.convert("I"))


def imagemath_float(self):
    return _Operand(self.im.convert("F"))


# logical
def imagemath_equal(self, other):
    return self.apply("eq", self, other, mode="I")


def imagemath_notequal(self, other):
    return self.apply("ne", self, other, mode="I")


def imagemath_min(self, other):
    return self.apply("min", self, other)


def imagemath_max(self, other):
    return self.apply("max", self, other)


def imagemath_convert(self, mode):
    return _Operand(self.im.convert(mode))


ops = {}
for k, v in list(globals().items()):
    if k[:10] == "imagemath_":
        ops[k[10:]] = v


def eval(expression, _dict={}, **kw):
    """
    Evaluates an image expression.

    :param expression: A string containing a Python-style expression.
    :param options: Values to add to the evaluation context.  You
                    can either use a dictionary, or one or more keyword
                    arguments.
    :return: The evaluated expression. This is usually an image object, but can
             also be an integer, a floating point value, or a pixel tuple,
             depending on the expression.
    """

    # build execution namespace
    args = ops.copy()
    args.update(_dict)
    args.update(kw)
    for k, v in list(args.items()):
        if hasattr(v, "im"):
            args[k] = _Operand(v)

    compiled_code = compile(expression, "<string>", "eval")

    def scan(code):
        for const in code.co_consts:
            if type(const) == type(compiled_code):
                scan(const)

        for name in code.co_names:
            if name not in args and name != "abs":
                msg = f"'{name}' not allowed"
                raise ValueError(msg)

    scan(compiled_code)
    out = builtins.eval(expression, {"__builtins": {"abs": abs}}, args)
    try:
        return out.im
    except AttributeError:
        return out

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import pytest
from itertools import chain

def test_packbits():
    # Copied from the docstring.
    a = [[[1, 0, 1], [0, 1, 0]],
         [[1, 1, 0], [0, 0, 1]]]
    for dt in '?bBhHiIlLqQ':
        arr = np.array(a, dtype=dt)
        b = np.packbits(arr, axis=-1)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, np.array([[[160], [64]], [[192], [32]]]))

    assert_raises(TypeError, np.packbits, np.array(a, dtype=float))


def test_packbits_empty():
    shapes = [
        (0,), (10, 20, 0), (10, 0, 20), (0, 10, 20), (20, 0, 0), (0, 20, 0),
        (0, 0, 20), (0, 0, 0),
    ]
    for dt in '?bBhHiIlLqQ':
        for shape in shapes:
            a = np.empty(shape, dtype=dt)
            b = np.packbits(a)
            assert_equal(b.dtype, np.uint8)
            assert_equal(b.shape, (0,))


def test_packbits_empty_with_axis():
    # Original shapes and lists of packed shapes for different axes.
    shapes = [
        ((0,), [(0,)]),
        ((10, 20, 0), [(2, 20, 0), (10, 3, 0), (10, 20, 0)]),
        ((10, 0, 20), [(2, 0, 20), (10, 0, 20), (10, 0, 3)]),
        ((0, 10, 20), [(0, 10, 20), (0, 2, 20), (0, 10, 3)]),
        ((20, 0, 0), [(3, 0, 0), (20, 0, 0), (20, 0, 0)]),
        ((0, 20, 0), [(0, 20, 0), (0, 3, 0), (0, 20, 0)]),
        ((0, 0, 20), [(0, 0, 20), (0, 0, 20), (0, 0, 3)]),
        ((0, 0, 0), [(0, 0, 0), (0, 0, 0), (0, 0, 0)]),
    ]
    for dt in '?bBhHiIlLqQ':
        for in_shape, out_shapes in shapes:
            for ax, out_shape in enumerate(out_shapes):
                a = np.empty(in_shape, dtype=dt)
                b = np.packbits(a, axis=ax)
                assert_equal(b.dtype, np.uint8)
                assert_equal(b.shape, out_shape)

@pytest.mark.parametrize('bitorder', ('little', 'big'))
def test_packbits_large(bitorder):
    # test data large enough for 16 byte vectorization
    a = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
                  0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
                  1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                  1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
                  1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
                  1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
                  1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
                  0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1,
                  1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
                  1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
                  1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
                  0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
                  1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
                  1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0,
                  1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    a = a.repeat(3)
    for dtype in '?bBhHiIlLqQ':
        arr = np.array(a, dtype=dtype)
        b = np.packbits(arr, axis=None, bitorder=bitorder)
        assert_equal(b.dtype, np.uint8)
        r = [252, 127, 192, 3, 254, 7, 252, 0, 7, 31, 240, 0, 28, 1, 255, 252,
             113, 248, 3, 255, 192, 28, 15, 192, 28, 126, 0, 224, 127, 255,
             227, 142, 7, 31, 142, 63, 28, 126, 56, 227, 240, 0, 227, 128, 63,
             224, 14, 56, 252, 112, 56, 255, 241, 248, 3, 240, 56, 224, 112,
             63, 255, 255, 199, 224, 14, 0, 31, 143, 192, 3, 255, 199, 0, 1,
             255, 224, 1, 255, 252, 126, 63, 0, 1, 192, 252, 14, 63, 0, 15,
             199, 252, 113, 255, 3, 128, 56, 252, 14, 7, 0, 113, 255, 255, 142, 56, 227,
             129, 248, 227, 129, 199, 31, 128]
        if bitorder == 'big':
            assert_array_equal(b, r)
        # equal for size being multiple of 8
        assert_array_equal(np.unpackbits(b, bitorder=bitorder)[:-4], a)

        # check last byte of different remainders (16 byte vectorization)
        b = [np.packbits(arr[:-i], axis=None)[-1] for i in range(1, 16)]
        assert_array_equal(b, [128, 128, 128, 31, 30, 28, 24, 16, 0, 0, 0, 199,
                               198, 196, 192])


        arr = arr.reshape(36, 25)
        b = np.packbits(arr, axis=0)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, [[190, 186, 178, 178, 150, 215, 87, 83, 83, 195,
                                199, 206, 204, 204, 140, 140, 136, 136, 8, 40, 105,
                                107, 75, 74, 88],
                               [72, 216, 248, 241, 227, 195, 202, 90, 90, 83,
                                83, 119, 127, 109, 73, 64, 208, 244, 189, 45,
                                41, 104, 122, 90, 18],
                               [113, 120, 248, 216, 152, 24, 60, 52, 182, 150,
                                150, 150, 146, 210, 210, 246, 255, 255, 223,
                                151, 21, 17, 17, 131, 163],
                               [214, 210, 210, 64, 68, 5, 5, 1, 72, 88, 92,
                                92, 78, 110, 39, 181, 149, 220, 222, 218, 218,
                                202, 234, 170, 168],
                               [0, 128, 128, 192, 80, 112, 48, 160, 160, 224,
                                240, 208, 144, 128, 160, 224, 240, 208, 144,
                                144, 176, 240, 224, 192, 128]])

        b = np.packbits(arr, axis=1)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, [[252, 127, 192,   0],
                               [  7, 252,  15, 128],
                               [240,   0,  28,   0],
                               [255, 128,   0, 128],
                               [192,  31, 255, 128],
                               [142,  63,   0,   0],
                               [255, 240,   7,   0],
                               [  7, 224,  14,   0],
                               [126,   0, 224,   0],
                               [255, 255, 199,   0],
                               [ 56,  28, 126,   0],
                               [113, 248, 227, 128],
                               [227, 142,  63,   0],
                               [  0,  28, 112,   0],
                               [ 15, 248,   3, 128],
                               [ 28, 126,  56,   0],
                               [ 56, 255, 241, 128],
                               [240,   7, 224,   0],
                               [227, 129, 192, 128],
                               [255, 255, 254,   0],
                               [126,   0, 224,   0],
                               [  3, 241, 248,   0],
                               [  0, 255, 241, 128],
                               [128,   0, 255, 128],
                               [224,   1, 255, 128],
                               [248, 252, 126,   0],
                               [  0,   7,   3, 128],
                               [224, 113, 248,   0],
                               [  0, 252, 127, 128],
                               [142,  63, 224,   0],
                               [224,  14,  63,   0],
                               [  7,   3, 128,   0],
                               [113, 255, 255, 128],
                               [ 28, 113, 199,   0],
                               [  7, 227, 142,   0],
                               [ 14,  56, 252,   0]])

        arr = arr.T.copy()
        b = np.packbits(arr, axis=0)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, [[252, 7, 240, 255, 192, 142, 255, 7, 126, 255,
                                56, 113, 227, 0, 15, 28, 56, 240, 227, 255,
                                126, 3, 0, 128, 224, 248, 0, 224, 0, 142, 224,
                                7, 113, 28, 7, 14],
                                [127, 252, 0, 128, 31, 63, 240, 224, 0, 255,
                                 28, 248, 142, 28, 248, 126, 255, 7, 129, 255,
                                 0, 241, 255, 0, 1, 252, 7, 113, 252, 63, 14,
                                 3, 255, 113, 227, 56],
                                [192, 15, 28, 0, 255, 0, 7, 14, 224, 199, 126,
                                 227, 63, 112, 3, 56, 241, 224, 192, 254, 224,
                                 248, 241, 255, 255, 126, 3, 248, 127, 224, 63,
                                 128, 255, 199, 142, 252],
                                [0, 128, 0, 128, 128, 0, 0, 0, 0, 0, 0, 128, 0,
                                 0, 128, 0, 128, 0, 128, 0, 0, 0, 128, 128,
                                 128, 0, 128, 0, 128, 0, 0, 0, 128, 0, 0, 0]])

        b = np.packbits(arr, axis=1)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, [[190,  72, 113, 214,   0],
                               [186, 216, 120, 210, 128],
                               [178, 248, 248, 210, 128],
                               [178, 241, 216,  64, 192],
                               [150, 227, 152,  68,  80],
                               [215, 195,  24,   5, 112],
                               [ 87, 202,  60,   5,  48],
                               [ 83,  90,  52,   1, 160],
                               [ 83,  90, 182,  72, 160],
                               [195,  83, 150,  88, 224],
                               [199,  83, 150,  92, 240],
                               [206, 119, 150,  92, 208],
                               [204, 127, 146,  78, 144],
                               [204, 109, 210, 110, 128],
                               [140,  73, 210,  39, 160],
                               [140,  64, 246, 181, 224],
                               [136, 208, 255, 149, 240],
                               [136, 244, 255, 220, 208],
                               [  8, 189, 223, 222, 144],
                               [ 40,  45, 151, 218, 144],
                               [105,  41,  21, 218, 176],
                               [107, 104,  17, 202, 240],
                               [ 75, 122,  17, 234, 224],
                               [ 74,  90, 131, 170, 192],
                               [ 88,  18, 163, 168, 128]])


    # result is the same if input is multiplied with a nonzero value
    for dtype in 'bBhHiIlLqQ':
        arr = np.array(a, dtype=dtype)
        rnd = np.random.randint(low=np.iinfo(dtype).min,
                                high=np.iinfo(dtype).max, size=arr.size,
                                dtype=dtype)
        rnd[rnd == 0] = 1
        arr *= rnd.astype(dtype)
        b = np.packbits(arr, axis=-1)
        assert_array_equal(np.unpackbits(b)[:-4], a)

    assert_raises(TypeError, np.packbits, np.array(a, dtype=float))


def test_packbits_very_large():
    # test some with a larger arrays gh-8637
    # code is covered earlier but larger array makes crash on bug more likely
    for s in range(950, 1050):
        for dt in '?bBhHiIlLqQ':
            x = np.ones((200, s), dtype=bool)
            np.packbits(x, axis=1)


def test_unpackbits():
    # Copied from the docstring.
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    b = np.unpackbits(a, axis=1)
    assert_equal(b.dtype, np.uint8)
    assert_array_equal(b, np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1],
                                    [0, 0, 0, 1, 0, 1, 1, 1]]))

def test_pack_unpack_order():
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    b = np.unpackbits(a, axis=1)
    assert_equal(b.dtype, np.uint8)
    b_little = np.unpackbits(a, axis=1, bitorder='little')
    b_big = np.unpackbits(a, axis=1, bitorder='big')
    assert_array_equal(b, b_big)
    assert_array_equal(a, np.packbits(b_little, axis=1, bitorder='little'))
    assert_array_equal(b[:,::-1], b_little)
    assert_array_equal(a, np.packbits(b_big, axis=1, bitorder='big'))
    assert_raises(ValueError, np.unpackbits, a, bitorder='r')
    assert_raises(TypeError, np.unpackbits, a, bitorder=10)



def test_unpackbits_empty():
    a = np.empty((0,), dtype=np.uint8)
    b = np.unpackbits(a)
    assert_equal(b.dtype, np.uint8)
    assert_array_equal(b, np.empty((0,)))


def test_unpackbits_empty_with_axis():
    # Lists of packed shapes for different axes and unpacked shapes.
    shapes = [
        ([(0,)], (0,)),
        ([(2, 24, 0), (16, 3, 0), (16, 24, 0)], (16, 24, 0)),
        ([(2, 0, 24), (16, 0, 24), (16, 0, 3)], (16, 0, 24)),
        ([(0, 16, 24), (0, 2, 24), (0, 16, 3)], (0, 16, 24)),
        ([(3, 0, 0), (24, 0, 0), (24, 0, 0)], (24, 0, 0)),
        ([(0, 24, 0), (0, 3, 0), (0, 24, 0)], (0, 24, 0)),
        ([(0, 0, 24), (0, 0, 24), (0, 0, 3)], (0, 0, 24)),
        ([(0, 0, 0), (0, 0, 0), (0, 0, 0)], (0, 0, 0)),
    ]
    for in_shapes, out_shape in shapes:
        for ax, in_shape in enumerate(in_shapes):
            a = np.empty(in_shape, dtype=np.uint8)
            b = np.unpackbits(a, axis=ax)
            assert_equal(b.dtype, np.uint8)
            assert_equal(b.shape, out_shape)


def test_unpackbits_large():
    # test all possible numbers via comparison to already tested packbits
    d = np.arange(277, dtype=np.uint8)
    assert_array_equal(np.packbits(np.unpackbits(d)), d)
    assert_array_equal(np.packbits(np.unpackbits(d[::2])), d[::2])
    d = np.tile(d, (3, 1))
    assert_array_equal(np.packbits(np.unpackbits(d, axis=1), axis=1), d)
    d = d.T.copy()
    assert_array_equal(np.packbits(np.unpackbits(d, axis=0), axis=0), d)


class TestCount():
    x = np.array([
        [1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
    ], dtype=np.uint8)
    padded1 = np.zeros(57, dtype=np.uint8)
    padded1[:49] = x.ravel()
    padded1b = np.zeros(57, dtype=np.uint8)
    padded1b[:49] = x[::-1].copy().ravel()
    padded2 = np.zeros((9, 9), dtype=np.uint8)
    padded2[:7, :7] = x

    @pytest.mark.parametrize('bitorder', ('little', 'big'))
    @pytest.mark.parametrize('count', chain(range(58), range(-1, -57, -1)))
    def test_roundtrip(self, bitorder, count):
        if count < 0:
            # one extra zero of padding
            cutoff = count - 1
        else:
            cutoff = count
        # test complete invertibility of packbits and unpackbits with count
        packed = np.packbits(self.x, bitorder=bitorder)
        unpacked = np.unpackbits(packed, count=count, bitorder=bitorder)
        assert_equal(unpacked.dtype, np.uint8)
        assert_array_equal(unpacked, self.padded1[:cutoff])

    @pytest.mark.parametrize('kwargs', [
                    {}, {'count': None},
                    ])
    def test_count(self, kwargs):
        packed = np.packbits(self.x)
        unpacked = np.unpackbits(packed, **kwargs)
        assert_equal(unpacked.dtype, np.uint8)
        assert_array_equal(unpacked, self.padded1[:-1])

    @pytest.mark.parametrize('bitorder', ('little', 'big'))
    # delta==-1 when count<0 because one extra zero of padding
    @pytest.mark.parametrize('count', chain(range(8), range(-1, -9, -1)))
    def test_roundtrip_axis(self, bitorder, count):
        if count < 0:
            # one extra zero of padding
            cutoff = count - 1
        else:
            cutoff = count
        packed0 = np.packbits(self.x, axis=0, bitorder=bitorder)
        unpacked0 = np.unpackbits(packed0, axis=0, count=count,
                                  bitorder=bitorder)
        assert_equal(unpacked0.dtype, np.uint8)
        assert_array_equal(unpacked0, self.padded2[:cutoff, :self.x.shape[1]])

        packed1 = np.packbits(self.x, axis=1, bitorder=bitorder)
        unpacked1 = np.unpackbits(packed1, axis=1, count=count,
                                  bitorder=bitorder)
        assert_equal(unpacked1.dtype, np.uint8)
        assert_array_equal(unpacked1, self.padded2[:self.x.shape[0], :cutoff])

    @pytest.mark.parametrize('kwargs', [
                    {}, {'count': None},
                    {'bitorder' : 'little'},
                    {'bitorder': 'little', 'count': None},
                    {'bitorder' : 'big'},
                    {'bitorder': 'big', 'count': None},
                    ])
    def test_axis_count(self, kwargs):
        packed0 = np.packbits(self.x, axis=0)
        unpacked0 = np.unpackbits(packed0, axis=0, **kwargs)
        assert_equal(unpacked0.dtype, np.uint8)
        if kwargs.get('bitorder', 'big') == 'big':
            assert_array_equal(unpacked0, self.padded2[:-1, :self.x.shape[1]])
        else:
            assert_array_equal(unpacked0[::-1, :], self.padded2[:-1, :self.x.shape[1]])

        packed1 = np.packbits(self.x, axis=1)
        unpacked1 = np.unpackbits(packed1, axis=1, **kwargs)
        assert_equal(unpacked1.dtype, np.uint8)
        if kwargs.get('bitorder', 'big') == 'big':
            assert_array_equal(unpacked1, self.padded2[:self.x.shape[0], :-1])
        else:
            assert_array_equal(unpacked1[:, ::-1], self.padded2[:self.x.shape[0], :-1])

    def test_bad_count(self):
        packed0 = np.packbits(self.x, axis=0)
        assert_raises(ValueError, np.unpackbits, packed0, axis=0, count=-9)
        packed1 = np.packbits(self.x, axis=1)
        assert_raises(ValueError, np.unpackbits, packed1, axis=1, count=-9)
        packed = np.packbits(self.x)
        assert_raises(ValueError, np.unpackbits, packed, count=-57)

import pytest

import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_equal

def buffer_length(arr):
    if isinstance(arr, str):
        if not arr:
            charmax = 0
        else:
            charmax = max([ord(c) for c in arr])
        if charmax < 256:
            size = 1
        elif charmax < 65536:
            size = 2
        else:
            size = 4
        return size * len(arr)
    v = memoryview(arr)
    if v.shape is None:
        return len(v) * v.itemsize
    else:
        return np.prod(v.shape) * v.itemsize


# In both cases below we need to make sure that the byte swapped value (as
# UCS4) is still a valid unicode:
# Value that can be represented in UCS2 interpreters
ucs2_value = '\u0900'
# Value that cannot be represented in UCS2 interpreters (but can in UCS4)
ucs4_value = '\U00100900'


def test_string_cast():
    str_arr = np.array(["1234", "1234\0\0"], dtype='S')
    uni_arr1 = str_arr.astype('>U')
    uni_arr2 = str_arr.astype('<U')

    with pytest.warns(FutureWarning):
        assert str_arr != uni_arr1
    with pytest.warns(FutureWarning):
        assert str_arr != uni_arr2

    assert_array_equal(uni_arr1, uni_arr2)


############################################################
#    Creation tests
############################################################

class CreateZeros:
    """Check the creation of zero-valued arrays"""

    def content_check(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        assert_(buffer_length(ua) == nbytes)
        # Small check that data in array element is ok
        assert_(ua_scalar == '')
        # Encode to ascii and double check
        assert_(ua_scalar.encode('ascii') == b'')
        # Check buffer lengths for scalars
        assert_(buffer_length(ua_scalar) == 0)

    def test_zeros0D(self):
        # Check creation of 0-dimensional objects
        ua = np.zeros((), dtype='U%s' % self.ulen)
        self.content_check(ua, ua[()], 4*self.ulen)

    def test_zerosSD(self):
        # Check creation of single-dimensional objects
        ua = np.zeros((2,), dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0], 4*self.ulen*2)
        self.content_check(ua, ua[1], 4*self.ulen*2)

    def test_zerosMD(self):
        # Check creation of multi-dimensional objects
        ua = np.zeros((2, 3, 4), dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0, 0, 0], 4*self.ulen*2*3*4)
        self.content_check(ua, ua[-1, -1, -1], 4*self.ulen*2*3*4)


class TestCreateZeros_1(CreateZeros):
    """Check the creation of zero-valued arrays (size 1)"""
    ulen = 1


class TestCreateZeros_2(CreateZeros):
    """Check the creation of zero-valued arrays (size 2)"""
    ulen = 2


class TestCreateZeros_1009(CreateZeros):
    """Check the creation of zero-valued arrays (size 1009)"""
    ulen = 1009


class CreateValues:
    """Check the creation of unicode arrays with values"""

    def content_check(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        assert_(buffer_length(ua) == nbytes)
        # Small check that data in array element is ok
        assert_(ua_scalar == self.ucs_value*self.ulen)
        # Encode to UTF-8 and double check
        assert_(ua_scalar.encode('utf-8') ==
                        (self.ucs_value*self.ulen).encode('utf-8'))
        # Check buffer lengths for scalars
        if self.ucs_value == ucs4_value:
            # In UCS2, the \U0010FFFF will be represented using a
            # surrogate *pair*
            assert_(buffer_length(ua_scalar) == 2*2*self.ulen)
        else:
            # In UCS2, the \uFFFF will be represented using a
            # regular 2-byte word
            assert_(buffer_length(ua_scalar) == 2*self.ulen)

    def test_values0D(self):
        # Check creation of 0-dimensional objects with values
        ua = np.array(self.ucs_value*self.ulen, dtype='U%s' % self.ulen)
        self.content_check(ua, ua[()], 4*self.ulen)

    def test_valuesSD(self):
        # Check creation of single-dimensional objects with values
        ua = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0], 4*self.ulen*2)
        self.content_check(ua, ua[1], 4*self.ulen*2)

    def test_valuesMD(self):
        # Check creation of multi-dimensional objects with values
        ua = np.array([[[self.ucs_value*self.ulen]*2]*3]*4, dtype='U%s' % self.ulen)
        self.content_check(ua, ua[0, 0, 0], 4*self.ulen*2*3*4)
        self.content_check(ua, ua[-1, -1, -1], 4*self.ulen*2*3*4)


class TestCreateValues_1_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value


class TestCreateValues_1_UCS4(CreateValues):
    """Check the creation of valued arrays (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value


class TestCreateValues_2_UCS2(CreateValues):
    """Check the creation of valued arrays (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value


class TestCreateValues_2_UCS4(CreateValues):
    """Check the creation of valued arrays (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value


class TestCreateValues_1009_UCS2(CreateValues):
    """Check the creation of valued arrays (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value


class TestCreateValues_1009_UCS4(CreateValues):
    """Check the creation of valued arrays (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value


############################################################
#    Assignment tests
############################################################

class AssignValues:
    """Check the assignment of unicode arrays with values"""

    def content_check(self, ua, ua_scalar, nbytes):

        # Check the length of the unicode base type
        assert_(int(ua.dtype.str[2:]) == self.ulen)
        # Check the length of the data buffer
        assert_(buffer_length(ua) == nbytes)
        # Small check that data in array element is ok
        assert_(ua_scalar == self.ucs_value*self.ulen)
        # Encode to UTF-8 and double check
        assert_(ua_scalar.encode('utf-8') ==
                        (self.ucs_value*self.ulen).encode('utf-8'))
        # Check buffer lengths for scalars
        if self.ucs_value == ucs4_value:
            # In UCS2, the \U0010FFFF will be represented using a
            # surrogate *pair*
            assert_(buffer_length(ua_scalar) == 2*2*self.ulen)
        else:
            # In UCS2, the \uFFFF will be represented using a
            # regular 2-byte word
            assert_(buffer_length(ua_scalar) == 2*self.ulen)

    def test_values0D(self):
        # Check assignment of 0-dimensional objects with values
        ua = np.zeros((), dtype='U%s' % self.ulen)
        ua[()] = self.ucs_value*self.ulen
        self.content_check(ua, ua[()], 4*self.ulen)

    def test_valuesSD(self):
        # Check assignment of single-dimensional objects with values
        ua = np.zeros((2,), dtype='U%s' % self.ulen)
        ua[0] = self.ucs_value*self.ulen
        self.content_check(ua, ua[0], 4*self.ulen*2)
        ua[1] = self.ucs_value*self.ulen
        self.content_check(ua, ua[1], 4*self.ulen*2)

    def test_valuesMD(self):
        # Check assignment of multi-dimensional objects with values
        ua = np.zeros((2, 3, 4), dtype='U%s' % self.ulen)
        ua[0, 0, 0] = self.ucs_value*self.ulen
        self.content_check(ua, ua[0, 0, 0], 4*self.ulen*2*3*4)
        ua[-1, -1, -1] = self.ucs_value*self.ulen
        self.content_check(ua, ua[-1, -1, -1], 4*self.ulen*2*3*4)


class TestAssignValues_1_UCS2(AssignValues):
    """Check the assignment of valued arrays (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value


class TestAssignValues_1_UCS4(AssignValues):
    """Check the assignment of valued arrays (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value


class TestAssignValues_2_UCS2(AssignValues):
    """Check the assignment of valued arrays (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value


class TestAssignValues_2_UCS4(AssignValues):
    """Check the assignment of valued arrays (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value


class TestAssignValues_1009_UCS2(AssignValues):
    """Check the assignment of valued arrays (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value


class TestAssignValues_1009_UCS4(AssignValues):
    """Check the assignment of valued arrays (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value


############################################################
#    Byteorder tests
############################################################

class ByteorderValues:
    """Check the byteorder of unicode arrays in round-trip conversions"""

    def test_values0D(self):
        # Check byteorder of 0-dimensional objects
        ua = np.array(self.ucs_value*self.ulen, dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        # This changes the interpretation of the data region (but not the
        #  actual data), therefore the returned scalars are not
        #  the same (they are byte-swapped versions of each other).
        assert_(ua[()] != ua2[()])
        ua3 = ua2.newbyteorder()
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def test_valuesSD(self):
        # Check byteorder of single-dimensional objects
        ua = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        assert_((ua != ua2).all())
        assert_(ua[-1] != ua2[-1])
        ua3 = ua2.newbyteorder()
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def test_valuesMD(self):
        # Check byteorder of multi-dimensional objects
        ua = np.array([[[self.ucs_value*self.ulen]*2]*3]*4,
                      dtype='U%s' % self.ulen)
        ua2 = ua.newbyteorder()
        assert_((ua != ua2).all())
        assert_(ua[-1, -1, -1] != ua2[-1, -1, -1])
        ua3 = ua2.newbyteorder()
        # Arrays must be equal after the round-trip
        assert_equal(ua, ua3)

    def test_values_cast(self):
        # Check byteorder of when casting the array for a strided and
        # contiguous array:
        test1 = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        test2 = np.repeat(test1, 2)[::2]
        for ua in (test1, test2):
            ua2 = ua.astype(dtype=ua.dtype.newbyteorder())
            assert_((ua == ua2).all())
            assert_(ua[-1] == ua2[-1])
            ua3 = ua2.astype(dtype=ua.dtype)
            # Arrays must be equal after the round-trip
            assert_equal(ua, ua3)

    def test_values_updowncast(self):
        # Check byteorder of when casting the array to a longer and shorter
        # string length for strided and contiguous arrays
        test1 = np.array([self.ucs_value*self.ulen]*2, dtype='U%s' % self.ulen)
        test2 = np.repeat(test1, 2)[::2]
        for ua in (test1, test2):
            # Cast to a longer type with zero padding
            longer_type = np.dtype('U%s' % (self.ulen+1)).newbyteorder()
            ua2 = ua.astype(dtype=longer_type)
            assert_((ua == ua2).all())
            assert_(ua[-1] == ua2[-1])
            # Cast back again with truncating:
            ua3 = ua2.astype(dtype=ua.dtype)
            # Arrays must be equal after the round-trip
            assert_equal(ua, ua3)


class TestByteorder_1_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 1, UCS2 values)"""
    ulen = 1
    ucs_value = ucs2_value


class TestByteorder_1_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 1, UCS4 values)"""
    ulen = 1
    ucs_value = ucs4_value


class TestByteorder_2_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 2, UCS2 values)"""
    ulen = 2
    ucs_value = ucs2_value


class TestByteorder_2_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 2, UCS4 values)"""
    ulen = 2
    ucs_value = ucs4_value


class TestByteorder_1009_UCS2(ByteorderValues):
    """Check the byteorder in unicode (size 1009, UCS2 values)"""
    ulen = 1009
    ucs_value = ucs2_value


class TestByteorder_1009_UCS4(ByteorderValues):
    """Check the byteorder in unicode (size 1009, UCS4 values)"""
    ulen = 1009
    ucs_value = ucs4_value

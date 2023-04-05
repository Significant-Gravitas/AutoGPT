from cpython cimport PyBytes_AsString


#from cpython cimport PyByteArray_AsString # cython still not exports that
cdef extern from "Python.h":
    char* PyByteArray_AsString(bytearray ba) except NULL

from libc.stdint cimport uint32_t, uint64_t, uintmax_t


def _websocket_mask_cython(object mask, object data):
    """Note, this function mutates its `data` argument
    """
    cdef:
        Py_ssize_t data_len, i
        # bit operations on signed integers are implementation-specific
        unsigned char * in_buf
        const unsigned char * mask_buf
        uint32_t uint32_msk
        uint64_t uint64_msk

    assert len(mask) == 4

    if not isinstance(mask, bytes):
        mask = bytes(mask)

    if isinstance(data, bytearray):
        data = <bytearray>data
    else:
        data = bytearray(data)

    data_len = len(data)
    in_buf = <unsigned char*>PyByteArray_AsString(data)
    mask_buf = <const unsigned char*>PyBytes_AsString(mask)
    uint32_msk = (<uint32_t*>mask_buf)[0]

    # TODO: align in_data ptr to achieve even faster speeds
    # does it need in python ?! malloc() always aligns to sizeof(long) bytes

    if sizeof(size_t) >= 8:
        uint64_msk = uint32_msk
        uint64_msk = (uint64_msk << 32) | uint32_msk

        while data_len >= 8:
            (<uint64_t*>in_buf)[0] ^= uint64_msk
            in_buf += 8
            data_len -= 8


    while data_len >= 4:
        (<uint32_t*>in_buf)[0] ^= uint32_msk
        in_buf += 4
        data_len -= 4

    for i in range(0, data_len):
        in_buf[i] ^= mask_buf[i]

import sys
import os
import mmap
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile

from numpy import (
    memmap, sum, average, product, ndarray, isscalar, add, subtract, multiply)

from numpy import arange, allclose, asarray
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, suppress_warnings, IS_PYPY,
    break_cycles
    )

class TestMemmap:
    def setup_method(self):
        self.tmpfp = NamedTemporaryFile(prefix='mmap')
        self.shape = (3, 4)
        self.dtype = 'float32'
        self.data = arange(12, dtype=self.dtype)
        self.data.resize(self.shape)

    def teardown_method(self):
        self.tmpfp.close()
        self.data = None
        if IS_PYPY:
            break_cycles()
            break_cycles()

    def test_roundtrip(self):
        # Write data to file
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        del fp  # Test __del__ machinery, which handles cleanup

        # Read data back from file
        newfp = memmap(self.tmpfp, dtype=self.dtype, mode='r',
                       shape=self.shape)
        assert_(allclose(self.data, newfp))
        assert_array_equal(self.data, newfp)
        assert_equal(newfp.flags.writeable, False)

    def test_open_with_filename(self, tmp_path):
        tmpname = tmp_path / 'mmap'
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        fp[:] = self.data[:]
        del fp

    def test_unnamed_file(self):
        with TemporaryFile() as f:
            fp = memmap(f, dtype=self.dtype, shape=self.shape)
            del fp

    def test_attributes(self):
        offset = 1
        mode = "w+"
        fp = memmap(self.tmpfp, dtype=self.dtype, mode=mode,
                    shape=self.shape, offset=offset)
        assert_equal(offset, fp.offset)
        assert_equal(mode, fp.mode)
        del fp

    def test_filename(self, tmp_path):
        tmpname = tmp_path / "mmap"
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        abspath = Path(os.path.abspath(tmpname))
        fp[:] = self.data[:]
        assert_equal(abspath, fp.filename)
        b = fp[:1]
        assert_equal(abspath, b.filename)
        del b
        del fp

    def test_path(self, tmp_path):
        tmpname = tmp_path / "mmap"
        fp = memmap(Path(tmpname), dtype=self.dtype, mode='w+',
                       shape=self.shape)
        # os.path.realpath does not resolve symlinks on Windows
        # see: https://bugs.python.org/issue9949
        # use Path.resolve, just as memmap class does internally
        abspath = str(Path(tmpname).resolve())
        fp[:] = self.data[:]
        assert_equal(abspath, str(fp.filename.resolve()))
        b = fp[:1]
        assert_equal(abspath, str(b.filename.resolve()))
        del b
        del fp

    def test_filename_fileobj(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode="w+",
                    shape=self.shape)
        assert_equal(fp.filename, self.tmpfp.name)

    @pytest.mark.skipif(sys.platform == 'gnu0',
                        reason="Known to fail on hurd")
    def test_flush(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        assert_equal(fp[0], self.data[0])
        fp.flush()

    def test_del(self):
        # Make sure a view does not delete the underlying mmap
        fp_base = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp_base[0] = 5
        fp_view = fp_base[0:1]
        assert_equal(fp_view[0], 5)
        del fp_view
        # Should still be able to access and assign values after
        # deleting the view
        assert_equal(fp_base[0], 5)
        fp_base[0] = 6
        assert_equal(fp_base[0], 6)

    def test_arithmetic_drops_references(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        tmp = (fp + 10)
        if isinstance(tmp, memmap):
            assert_(tmp._mmap is not fp._mmap)

    def test_indexing_drops_references(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        tmp = fp[(1, 2), (2, 3)]
        if isinstance(tmp, memmap):
            assert_(tmp._mmap is not fp._mmap)

    def test_slicing_keeps_references(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        assert_(fp[:2, :2]._mmap is fp._mmap)

    def test_view(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        new1 = fp.view()
        new2 = new1.view()
        assert_(new1.base is fp)
        assert_(new2.base is fp)
        new_array = asarray(fp)
        assert_(new_array.base is fp)

    def test_ufunc_return_ndarray(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "np.average currently does not preserve")
            for unary_op in [sum, average, product]:
                result = unary_op(fp)
                assert_(isscalar(result))
                assert_(result.__class__ is self.data[0, 0].__class__)

                assert_(unary_op(fp, axis=0).__class__ is ndarray)
                assert_(unary_op(fp, axis=1).__class__ is ndarray)

        for binary_op in [add, subtract, multiply]:
            assert_(binary_op(fp, self.data).__class__ is ndarray)
            assert_(binary_op(self.data, fp).__class__ is ndarray)
            assert_(binary_op(fp, fp).__class__ is ndarray)

        fp += 1
        assert(fp.__class__ is memmap)
        add(fp, 1, out=fp)
        assert(fp.__class__ is memmap)

    def test_getitem(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        assert_(fp[1:, :-1].__class__ is memmap)
        # Fancy indexing returns a copy that is not memmapped
        assert_(fp[[0, 1]].__class__ is ndarray)

    def test_memmap_subclass(self):
        class MemmapSubClass(memmap):
            pass

        fp = MemmapSubClass(self.tmpfp, dtype=self.dtype, shape=self.shape)
        fp[:] = self.data

        # We keep previous behavior for subclasses of memmap, i.e. the
        # ufunc and __getitem__ output is never turned into a ndarray
        assert_(sum(fp, axis=0).__class__ is MemmapSubClass)
        assert_(sum(fp).__class__ is MemmapSubClass)
        assert_(fp[1:, :-1].__class__ is MemmapSubClass)
        assert(fp[[0, 1]].__class__ is MemmapSubClass)

    def test_mmap_offset_greater_than_allocation_granularity(self):
        size = 5 * mmap.ALLOCATIONGRANULARITY
        offset = mmap.ALLOCATIONGRANULARITY + 1
        fp = memmap(self.tmpfp, shape=size, mode='w+', offset=offset)
        assert_(fp.offset == offset)

    def test_no_shape(self):
        self.tmpfp.write(b'a'*16)
        mm = memmap(self.tmpfp, dtype='float64')
        assert_equal(mm.shape, (2,))

    def test_empty_array(self):
        # gh-12653
        with pytest.raises(ValueError, match='empty file'):
            memmap(self.tmpfp, shape=(0,4), mode='w+')

        self.tmpfp.write(b'\0')

        # ok now the file is not empty
        memmap(self.tmpfp, shape=(0,4), mode='w+')

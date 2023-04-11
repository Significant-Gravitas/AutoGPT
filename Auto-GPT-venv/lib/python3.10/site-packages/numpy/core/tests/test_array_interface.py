import sys
import pytest
import numpy as np
from numpy.testing import extbuild


@pytest.fixture
def get_module(tmp_path):
    """ Some codes to generate data and manage temporary buffers use when
    sharing with numpy via the array interface protocol.
    """

    if not sys.platform.startswith('linux'):
        pytest.skip('link fails on cygwin')

    prologue = '''
        #include <Python.h>
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        #include <stdio.h>
        #include <math.h>

        NPY_NO_EXPORT
        void delete_array_struct(PyObject *cap) {

            /* get the array interface structure */
            PyArrayInterface *inter = (PyArrayInterface*)
                PyCapsule_GetPointer(cap, NULL);

            /* get the buffer by which data was shared */
            double *ptr = (double*)PyCapsule_GetContext(cap);

            /* for the purposes of the regression test set the elements
               to nan */
            for (npy_intp i = 0; i < inter->shape[0]; ++i)
                ptr[i] = nan("");

            /* free the shared buffer */
            free(ptr);

            /* free the array interface structure */
            free(inter->shape);
            free(inter);

            fprintf(stderr, "delete_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)ptr);
        }
        '''

    functions = [
        ("new_array_struct", "METH_VARARGS", """

            long long n_elem = 0;
            double value = 0.0;

            if (!PyArg_ParseTuple(args, "Ld", &n_elem, &value)) {
                Py_RETURN_NONE;
            }

            /* allocate and initialize the data to share with numpy */
            long long n_bytes = n_elem*sizeof(double);
            double *data = (double*)malloc(n_bytes);

            if (!data) {
                PyErr_Format(PyExc_MemoryError,
                    "Failed to malloc %lld bytes", n_bytes);

                Py_RETURN_NONE;
            }

            for (long long i = 0; i < n_elem; ++i) {
                data[i] = value;
            }

            /* calculate the shape and stride */
            int nd = 1;

            npy_intp *ss = (npy_intp*)malloc(2*nd*sizeof(npy_intp));
            npy_intp *shape = ss;
            npy_intp *stride = ss + nd;

            shape[0] = n_elem;
            stride[0] = sizeof(double);

            /* construct the array interface */
            PyArrayInterface *inter = (PyArrayInterface*)
                malloc(sizeof(PyArrayInterface));

            memset(inter, 0, sizeof(PyArrayInterface));

            inter->two = 2;
            inter->nd = nd;
            inter->typekind = 'f';
            inter->itemsize = sizeof(double);
            inter->shape = shape;
            inter->strides = stride;
            inter->data = data;
            inter->flags = NPY_ARRAY_WRITEABLE | NPY_ARRAY_NOTSWAPPED |
                           NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS;

            /* package into a capsule */
            PyObject *cap = PyCapsule_New(inter, NULL, delete_array_struct);

            /* save the pointer to the data */
            PyCapsule_SetContext(cap, data);

            fprintf(stderr, "new_array_struct\\ncap = %ld inter = %ld"
                " ptr = %ld\\n", (long)cap, (long)inter, (long)data);

            return cap;
        """)
        ]

    more_init = "import_array();"

    try:
        import array_interface_testing
        return array_interface_testing
    except ImportError:
        pass

    # if it does not exist, build and load it
    return extbuild.build_and_import_extension('array_interface_testing',
                                               functions,
                                               prologue=prologue,
                                               include_dirs=[np.get_include()],
                                               build_dir=tmp_path,
                                               more_init=more_init)


@pytest.mark.slow
def test_cstruct(get_module):

    class data_source:
        """
        This class is for testing the timing of the PyCapsule destructor
        invoked when numpy release its reference to the shared data as part of
        the numpy array interface protocol. If the PyCapsule destructor is
        called early the shared data is freed and invalid memory accesses will
        occur.
        """

        def __init__(self, size, value):
            self.size = size
            self.value = value

        @property
        def __array_struct__(self):
            return get_module.new_array_struct(self.size, self.value)

    # write to the same stream as the C code
    stderr = sys.__stderr__

    # used to validate the shared data.
    expected_value = -3.1415
    multiplier = -10000.0

    # create some data to share with numpy via the array interface
    # assign the data an expected value.
    stderr.write(' ---- create an object to share data ---- \n')
    buf = data_source(256, expected_value)
    stderr.write(' ---- OK!\n\n')

    # share the data
    stderr.write(' ---- share data via the array interface protocol ---- \n')
    arr = np.array(buf, copy=False)
    stderr.write('arr.__array_interface___ = %s\n' % (
                 str(arr.__array_interface__)))
    stderr.write('arr.base = %s\n' % (str(arr.base)))
    stderr.write(' ---- OK!\n\n')

    # release the source of the shared data. this will not release the data
    # that was shared with numpy, that is done in the PyCapsule destructor.
    stderr.write(' ---- destroy the object that shared data ---- \n')
    buf = None
    stderr.write(' ---- OK!\n\n')

    # check that we got the expected data. If the PyCapsule destructor we
    # defined was prematurely called then this test will fail because our
    # destructor sets the elements of the array to NaN before free'ing the
    # buffer. Reading the values here may also cause a SEGV
    assert np.allclose(arr, expected_value)

    # read the data. If the PyCapsule destructor we defined was prematurely
    # called then reading the values here may cause a SEGV and will be reported
    # as invalid reads by valgrind
    stderr.write(' ---- read shared data ---- \n')
    stderr.write('arr = %s\n' % (str(arr)))
    stderr.write(' ---- OK!\n\n')

    # write to the shared buffer. If the shared data was prematurely deleted
    # this will may cause a SEGV and valgrind will report invalid writes
    stderr.write(' ---- modify shared data ---- \n')
    arr *= multiplier
    expected_value *= multiplier
    stderr.write('arr.__array_interface___ = %s\n' % (
                 str(arr.__array_interface__)))
    stderr.write('arr.base = %s\n' % (str(arr.base)))
    stderr.write(' ---- OK!\n\n')

    # read the data. If the shared data was prematurely deleted this
    # will may cause a SEGV and valgrind will report invalid reads
    stderr.write(' ---- read modified shared data ---- \n')
    stderr.write('arr = %s\n' % (str(arr)))
    stderr.write(' ---- OK!\n\n')

    # check that we got the expected data. If the PyCapsule destructor we
    # defined was prematurely called then this test will fail because our
    # destructor sets the elements of the array to NaN before free'ing the
    # buffer. Reading the values here may also cause a SEGV
    assert np.allclose(arr, expected_value)

    # free the shared data, the PyCapsule destructor should run here
    stderr.write(' ---- free shared data ---- \n')
    arr = None
    stderr.write(' ---- OK!\n\n')

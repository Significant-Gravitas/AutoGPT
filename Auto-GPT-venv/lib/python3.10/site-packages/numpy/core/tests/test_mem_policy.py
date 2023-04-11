import asyncio
import gc
import os
import pytest
import numpy as np
import threading
import warnings
from numpy.testing import extbuild, assert_warns, IS_WASM
import sys


@pytest.fixture
def get_module(tmp_path):
    """ Add a memory policy that returns a false pointer 64 bytes into the
    actual allocation, and fill the prefix with some text. Then check at each
    memory manipulation that the prefix exists, to make sure all alloc/realloc/
    free/calloc go via the functions here.
    """
    if sys.platform.startswith('cygwin'):
        pytest.skip('link fails on cygwin')
    if IS_WASM:
        pytest.skip("Can't build module inside Wasm")
    functions = [
        ("get_default_policy", "METH_NOARGS", """
             Py_INCREF(PyDataMem_DefaultHandler);
             return PyDataMem_DefaultHandler;
         """),
        ("set_secret_data_policy", "METH_NOARGS", """
             PyObject *secret_data =
                 PyCapsule_New(&secret_data_handler, "mem_handler", NULL);
             if (secret_data == NULL) {
                 return NULL;
             }
             PyObject *old = PyDataMem_SetHandler(secret_data);
             Py_DECREF(secret_data);
             return old;
         """),
        ("set_old_policy", "METH_O", """
             PyObject *old;
             if (args != NULL && PyCapsule_CheckExact(args)) {
                 old = PyDataMem_SetHandler(args);
             }
             else {
                 old = PyDataMem_SetHandler(NULL);
             }
             return old;
         """),
        ("get_array", "METH_NOARGS", """
            char *buf = (char *)malloc(20);
            npy_intp dims[1];
            dims[0] = 20;
            PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);
            return PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims, NULL,
                                        buf, NPY_ARRAY_WRITEABLE, NULL);
         """),
        ("set_own", "METH_O", """
            if (!PyArray_Check(args)) {
                PyErr_SetString(PyExc_ValueError,
                             "need an ndarray");
                return NULL;
            }
            PyArray_ENABLEFLAGS((PyArrayObject*)args, NPY_ARRAY_OWNDATA);
            // Maybe try this too?
            // PyArray_BASE(PyArrayObject *)args) = NULL;
            Py_RETURN_NONE;
         """),
        ("get_array_with_base", "METH_NOARGS", """
            char *buf = (char *)malloc(20);
            npy_intp dims[1];
            dims[0] = 20;
            PyArray_Descr *descr =  PyArray_DescrNewFromType(NPY_UINT8);
            PyObject *arr = PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims,
                                                 NULL, buf,
                                                 NPY_ARRAY_WRITEABLE, NULL);
            if (arr == NULL) return NULL;
            PyObject *obj = PyCapsule_New(buf, "buf capsule",
                                          (PyCapsule_Destructor)&warn_on_free);
            if (obj == NULL) {
                Py_DECREF(arr);
                return NULL;
            }
            if (PyArray_SetBaseObject((PyArrayObject *)arr, obj) < 0) {
                Py_DECREF(arr);
                Py_DECREF(obj);
                return NULL;
            }
            return arr;

         """),
    ]
    prologue = '''
        #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
        #include <numpy/arrayobject.h>
        /*
         * This struct allows the dynamic configuration of the allocator funcs
         * of the `secret_data_allocator`. It is provided here for
         * demonstration purposes, as a valid `ctx` use-case scenario.
         */
        typedef struct {
            void *(*malloc)(size_t);
            void *(*calloc)(size_t, size_t);
            void *(*realloc)(void *, size_t);
            void (*free)(void *);
        } SecretDataAllocatorFuncs;

        NPY_NO_EXPORT void *
        shift_alloc(void *ctx, size_t sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            char *real = (char *)funcs->malloc(sz + 64);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld", (unsigned long)sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void *
        shift_zero(void *ctx, size_t sz, size_t cnt) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            char *real = (char *)funcs->calloc(sz + 64, cnt);
            if (real == NULL) {
                return NULL;
            }
            snprintf(real, 64, "originally allocated %ld via zero",
                     (unsigned long)sz);
            return (void *)(real + 64);
        }
        NPY_NO_EXPORT void
        shift_free(void *ctx, void * p, npy_uintp sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            if (p == NULL) {
                return ;
            }
            char *real = (char *)p - 64;
            if (strncmp(real, "originally allocated", 20) != 0) {
                fprintf(stdout, "uh-oh, unmatched shift_free, "
                        "no appropriate prefix\\n");
                /* Make C runtime crash by calling free on the wrong address */
                funcs->free((char *)p + 10);
                /* funcs->free(real); */
            }
            else {
                npy_uintp i = (npy_uintp)atoi(real +20);
                if (i != sz) {
                    fprintf(stderr, "uh-oh, unmatched shift_free"
                            "(ptr, %ld) but allocated %ld\\n", sz, i);
                    /* This happens in some places, only print */
                    funcs->free(real);
                }
                else {
                    funcs->free(real);
                }
            }
        }
        NPY_NO_EXPORT void *
        shift_realloc(void *ctx, void * p, npy_uintp sz) {
            SecretDataAllocatorFuncs *funcs = (SecretDataAllocatorFuncs *)ctx;
            if (p != NULL) {
                char *real = (char *)p - 64;
                if (strncmp(real, "originally allocated", 20) != 0) {
                    fprintf(stdout, "uh-oh, unmatched shift_realloc\\n");
                    return realloc(p, sz);
                }
                return (void *)((char *)funcs->realloc(real, sz + 64) + 64);
            }
            else {
                char *real = (char *)funcs->realloc(p, sz + 64);
                if (real == NULL) {
                    return NULL;
                }
                snprintf(real, 64, "originally allocated "
                         "%ld  via realloc", (unsigned long)sz);
                return (void *)(real + 64);
            }
        }
        /* As an example, we use the standard {m|c|re}alloc/free funcs. */
        static SecretDataAllocatorFuncs secret_data_handler_ctx = {
            malloc,
            calloc,
            realloc,
            free
        };
        static PyDataMem_Handler secret_data_handler = {
            "secret_data_allocator",
            1,
            {
                &secret_data_handler_ctx, /* ctx */
                shift_alloc,              /* malloc */
                shift_zero,               /* calloc */
                shift_realloc,            /* realloc */
                shift_free                /* free */
            }
        };
        void warn_on_free(void *capsule) {
            PyErr_WarnEx(PyExc_UserWarning, "in warn_on_free", 1);
            void * obj = PyCapsule_GetPointer(capsule,
                                              PyCapsule_GetName(capsule));
            free(obj);
        };
        '''
    more_init = "import_array();"
    try:
        import mem_policy
        return mem_policy
    except ImportError:
        pass
    # if it does not exist, build and load it
    return extbuild.build_and_import_extension('mem_policy',
                                               functions,
                                               prologue=prologue,
                                               include_dirs=[np.get_include()],
                                               build_dir=tmp_path,
                                               more_init=more_init)


def test_set_policy(get_module):

    get_handler_name = np.core.multiarray.get_handler_name
    get_handler_version = np.core.multiarray.get_handler_version
    orig_policy_name = get_handler_name()

    a = np.arange(10).reshape((2, 5))  # a doesn't own its own data
    assert get_handler_name(a) is None
    assert get_handler_version(a) is None
    assert get_handler_name(a.base) == orig_policy_name
    assert get_handler_version(a.base) == 1

    orig_policy = get_module.set_secret_data_policy()

    b = np.arange(10).reshape((2, 5))  # b doesn't own its own data
    assert get_handler_name(b) is None
    assert get_handler_version(b) is None
    assert get_handler_name(b.base) == 'secret_data_allocator'
    assert get_handler_version(b.base) == 1

    if orig_policy_name == 'default_allocator':
        get_module.set_old_policy(None)  # tests PyDataMem_SetHandler(NULL)
        assert get_handler_name() == 'default_allocator'
    else:
        get_module.set_old_policy(orig_policy)
        assert get_handler_name() == orig_policy_name


def test_default_policy_singleton(get_module):
    get_handler_name = np.core.multiarray.get_handler_name

    # set the policy to default
    orig_policy = get_module.set_old_policy(None)

    assert get_handler_name() == 'default_allocator'

    # re-set the policy to default
    def_policy_1 = get_module.set_old_policy(None)

    assert get_handler_name() == 'default_allocator'

    # set the policy to original
    def_policy_2 = get_module.set_old_policy(orig_policy)

    # since default policy is a singleton,
    # these should be the same object
    assert def_policy_1 is def_policy_2 is get_module.get_default_policy()


def test_policy_propagation(get_module):
    # The memory policy goes hand-in-hand with flags.owndata

    class MyArr(np.ndarray):
        pass

    get_handler_name = np.core.multiarray.get_handler_name
    orig_policy_name = get_handler_name()
    a = np.arange(10).view(MyArr).reshape((2, 5))
    assert get_handler_name(a) is None
    assert a.flags.owndata is False

    assert get_handler_name(a.base) is None
    assert a.base.flags.owndata is False

    assert get_handler_name(a.base.base) == orig_policy_name
    assert a.base.base.flags.owndata is True


async def concurrent_context1(get_module, orig_policy_name, event):
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'
    else:
        get_module.set_old_policy(None)
        assert np.core.multiarray.get_handler_name() == 'default_allocator'
    event.set()


async def concurrent_context2(get_module, orig_policy_name, event):
    await event.wait()
    # the policy is not affected by changes in parallel contexts
    assert np.core.multiarray.get_handler_name() == orig_policy_name
    # change policy in the child context
    if orig_policy_name == 'default_allocator':
        get_module.set_secret_data_policy()
        assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'
    else:
        get_module.set_old_policy(None)
        assert np.core.multiarray.get_handler_name() == 'default_allocator'


async def async_test_context_locality(get_module):
    orig_policy_name = np.core.multiarray.get_handler_name()

    event = asyncio.Event()
    # the child contexts inherit the parent policy
    concurrent_task1 = asyncio.create_task(
        concurrent_context1(get_module, orig_policy_name, event))
    concurrent_task2 = asyncio.create_task(
        concurrent_context2(get_module, orig_policy_name, event))
    await concurrent_task1
    await concurrent_task2

    # the parent context is not affected by child policy changes
    assert np.core.multiarray.get_handler_name() == orig_policy_name


def test_context_locality(get_module):
    if (sys.implementation.name == 'pypy'
            and sys.pypy_version_info[:3] < (7, 3, 6)):
        pytest.skip('no context-locality support in PyPy < 7.3.6')
    asyncio.run(async_test_context_locality(get_module))


def concurrent_thread1(get_module, event):
    get_module.set_secret_data_policy()
    assert np.core.multiarray.get_handler_name() == 'secret_data_allocator'
    event.set()


def concurrent_thread2(get_module, event):
    event.wait()
    # the policy is not affected by changes in parallel threads
    assert np.core.multiarray.get_handler_name() == 'default_allocator'
    # change policy in the child thread
    get_module.set_secret_data_policy()


def test_thread_locality(get_module):
    orig_policy_name = np.core.multiarray.get_handler_name()

    event = threading.Event()
    # the child threads do not inherit the parent policy
    concurrent_task1 = threading.Thread(target=concurrent_thread1,
                                        args=(get_module, event))
    concurrent_task2 = threading.Thread(target=concurrent_thread2,
                                        args=(get_module, event))
    concurrent_task1.start()
    concurrent_task2.start()
    concurrent_task1.join()
    concurrent_task2.join()

    # the parent thread is not affected by child policy changes
    assert np.core.multiarray.get_handler_name() == orig_policy_name


@pytest.mark.slow
def test_new_policy(get_module):
    a = np.arange(10)
    orig_policy_name = np.core.multiarray.get_handler_name(a)

    orig_policy = get_module.set_secret_data_policy()

    b = np.arange(10)
    assert np.core.multiarray.get_handler_name(b) == 'secret_data_allocator'

    # test array manipulation. This is slow
    if orig_policy_name == 'default_allocator':
        # when the np.core.test tests recurse into this test, the
        # policy will be set so this "if" will be false, preventing
        # infinite recursion
        #
        # if needed, debug this by
        # - running tests with -- -s (to not capture stdout/stderr
        # - setting extra_argv=['-vv'] here
        assert np.core.test('full', verbose=2, extra_argv=['-vv'])
        # also try the ma tests, the pickling test is quite tricky
        assert np.ma.test('full', verbose=2, extra_argv=['-vv'])

    get_module.set_old_policy(orig_policy)

    c = np.arange(10)
    assert np.core.multiarray.get_handler_name(c) == orig_policy_name

@pytest.mark.xfail(sys.implementation.name == "pypy",
                   reason=("bad interaction between getenv and "
                           "os.environ inside pytest"))
@pytest.mark.parametrize("policy", ["0", "1", None])
def test_switch_owner(get_module, policy):
    a = get_module.get_array()
    assert np.core.multiarray.get_handler_name(a) is None
    get_module.set_own(a)
    oldval = os.environ.get('NUMPY_WARN_IF_NO_MEM_POLICY', None)
    if policy is None:
        if 'NUMPY_WARN_IF_NO_MEM_POLICY' in os.environ:
            os.environ.pop('NUMPY_WARN_IF_NO_MEM_POLICY')
    else:
        os.environ['NUMPY_WARN_IF_NO_MEM_POLICY'] = policy
    try:
        # The policy should be NULL, so we have to assume we can call
        # "free".  A warning is given if the policy == "1"
        if policy == "1":
            with assert_warns(RuntimeWarning) as w:
                del a
                gc.collect()
        else:
            del a
            gc.collect()

    finally:
        if oldval is None:
            if 'NUMPY_WARN_IF_NO_MEM_POLICY' in os.environ:
                os.environ.pop('NUMPY_WARN_IF_NO_MEM_POLICY')
        else:
            os.environ['NUMPY_WARN_IF_NO_MEM_POLICY'] = oldval

def test_owner_is_base(get_module):
    a = get_module.get_array_with_base()
    with pytest.warns(UserWarning, match='warn_on_free'):
        del a
        gc.collect()

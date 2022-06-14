#include <Python.h>
#include <stdio.h>
#include "arrayobject.h"

static char hello_docs[] = "helloworld(): Any message you want to put here!!\n";
static PyObject* hello(PyObject* self, PyObject *args) {
    PyObject *file = NULL;

    if (!PyArg_ParseTuple(args, "O", &file)) { return NULL; }

    PyObject *pystr = PyUnicode_FromString("example print from a C code\n");
    PyFile_WriteObject(pystr, file, Py_PRINT_RAW);

    return Py_BuildValue("");
}

static char cholesky_docs[] = "cholesky(np.array) -> np.array\n"
                              "\n"
                              "Cholesky decomposition of a positive semi-definite symmetric matrix.\n";
static PyObject *
cholesky(PyObject *self, PyObject *args)
{
    PyObject *X_object;
    PyArrayObject *L_array=NULL, *X_array=NULL;
    PyArrayIterObject *L_iter=NULL;

    if (!PyArg_ParseTuple(args, "O", &X_object)) {
        return NULL;
    }

    X_array = (PyArrayObject*) PyArray_FROM_OTF(X_object, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (X_array == NULL) { return NULL; }

    // get the shape of 0-th dimension of X_arrya
    const unsigned num_X = PyArray_DIM(X_array, 0);

    // create L_array from X_array as a prototype
#if NPY_API_VERSION >= 0x0000000c
    L_array = (PyArrayObject*) PyArray_NewLikeArray(X_array, NPY_KEEPORDER, NULL, 0);
#else
    L_array = (PyArrayObject*) PyArray_NewLikeArray(X_array, NPY_KEEPORDER, NULL, 0);
#endif

    if (L_array == NULL) { goto fail; }

    // iterate over the created numpy array
    L_iter = (PyArrayIterObject *) PyArray_IterNew(L_array);
    npy_intp *new_coordinates = (npy_intp []) {0, 1};
    PyArray_ITER_GOTO(L_iter, new_coordinates);
    double *L_dataptr = (double *)L_iter->dataptr;
    *L_dataptr = 1.0;

//    while(L_iter->index < L_iter->size) {
//        double *L_dataptr = (double *)L_iter->dataptr;
//        *L_dataptr = 1.0;
//        printf("Array element value = %f\n", *L_dataptr);
//        PyArray_ITER_NEXT(L_iter);
//    }

//    PyErr_SetString(PyExc_AttributeError, "first");
//    return NULL;

    // access the data
    Py_XDECREF(X_array);
    Py_XDECREF(L_iter);
    return (PyObject*) L_array;

  fail:
    Py_XDECREF(X_array);
    Py_XDECREF(L_array);
    Py_XDECREF(L_iter);

    return NULL;
}

static char example_wrapper_docs[] = "example_wrapper(): An example of C-numpy python binding.\n";
static PyObject *
example_wrapper(PyObject *dummy, PyObject *args)
{
    PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
    PyObject *arr1=NULL, *arr2=NULL, *oarr=NULL;

    if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2, &PyArray_Type, &out)) {
        return NULL;
    }

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) { return NULL; }

    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr2 == NULL) { goto fail; }

#if NPY_API_VERSION >= 0x0000000c
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY2);
#else
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
#endif
    if (oarr == NULL) goto fail;

    /* See: https://numpy.org/devdocs/reference/c-api/array.html */

    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */

    Py_DECREF(arr1);
    Py_DECREF(arr2);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_ResolveWritebackIfCopy(oarr);
#endif
    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
#if NPY_API_VERSION >= 0x0000000c
    PyArray_DiscardWritebackIfCopy(oarr);
#endif
    Py_XDECREF(oarr);
    return NULL;
}

static PyMethodDef biclustering_funcs[] = {
    {"hello", (PyCFunction)hello, METH_VARARGS, hello_docs},
    {"example_wrapper", (PyCFunction)example_wrapper, METH_VARARGS, example_wrapper_docs},
    {"cholesky", (PyCFunction)cholesky, METH_VARARGS, cholesky_docs},
    {NULL}
};

static char module_docs[] = "An algorithm for biclustering, based on sequential quadratic programming.\n"
                            "Removes most of the weights of a matrix, making it sparse.";
static struct PyModuleDef cModPyDem = {
    PyModuleDef_HEAD_INIT,
    "doubly_sparse_biclustering",
    module_docs,
    -1,
    biclustering_funcs
};

PyMODINIT_FUNC PyInit_biclustering_lasso(void) {
    // Call to import_array() is required, or we'll get a segfault at the first ndarray operation.
    // See: https://numpy.org/devdocs/user/c-info.how-to-extend.html
    // See: https://stackoverflow.com/questions/60748039/building-numpy-c-extension-segfault-when-calling-pyarray-from-otf
    import_array();

    return PyModule_Create(&cModPyDem);
};
#include <Python.h>

static PyObject* hello(PyObject* self, PyObject *args)
{
    PyObject *file = NULL;
    if (!PyArg_ParseTuple(args, "O", &file))
        return NULL;
    PyObject *pystr = PyUnicode_FromString("example print from a C code\n");
    PyFile_WriteObject(pystr, file, Py_PRINT_RAW);
   return Py_BuildValue("");
}

static char helloworld_docs[] =
   "helloworld(): Any message you want to put here!!\n";

static PyMethodDef helloworld_funcs[] = {
   {"hello", (PyCFunction)hello, METH_VARARGS, helloworld_docs},
   {NULL}
};

static struct PyModuleDef cModPyDem =
{
    PyModuleDef_HEAD_INIT,
    "helloworld",
    "Extension module example!",
    -1,
    helloworld_funcs
};

PyMODINIT_FUNC PyInit_hello(void)
{
    return PyModule_Create(&cModPyDem);
};
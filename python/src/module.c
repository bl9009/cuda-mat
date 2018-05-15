#include <Python.h>

#include "..\..\cuda\include\matrix.cuh"

#include "matrix.c"

static PyObject* pycudamat_from_2d(PyObject* self, PyObject* args)
{
    MatrixObject* mat = (MatrixObject*) PyObject_CallObject((PyObject*) &MatrixType, NULL);

    PyObject* arr;

    if (NULL != mat) {
        if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &arr)) {
            return NULL;
        }

        Py_ssize_t m = PyList_Size(arr);
        Py_ssize_t n = 0;

        if (m > 0) {
            PyObject* row = PyList_GetItem(arr, 0);

            if (PyList_Check(row)) {
                n = PyList_Size(row);
            }
        }

        mat->matrix = zeros(m, n);

        for (int i = 0; i < m; ++i) {
            PyObject* row = PyList_GetItem(arr, i);

            if (PyList_Check(row)) {
                Py_ssize_t n = PyList_Size(row);

                for (int j = 0; j < n; ++j) {
                    PyObject* value = PyList_GetItem(row, j);

                    double* c = NULL;

                    if (cell(mat->matrix, i, j, &c) == OUT_OF_BOUNDS) {
                        PyErr_SetString(PyExc_IndexError, "Index out of bounds!");

                        return NULL;
                    }

                    if (PyFloat_Check(value)) {
                        *c = PyFloat_AsDouble(value);
                    }
                    else {
                        *c = PyLong_AsDouble(value);
                    }
                }
            }
        }
    }

    return (PyObject*) mat;
}

static PyObject* pycudamat_zeros(PyObject* self, PyObject* args)
{
    MatrixObject* mat = (MatrixObject*) PyObject_CallObject((PyObject*) &MatrixType, NULL);

    int rows, cols;

    if (NULL != mat)
    {
        if (!PyArg_ParseTuple(args, "II", &rows, &cols)) {
            return NULL;
        }

        mat->matrix = zeros(rows, cols);
    }

    return (PyObject*) mat;
}

static PyObject* pycudamat_test(PyObject* self, PyObject* args)
{
    return (PyObject*) "Hello PyCUDA-Mat";
}

static PyMethodDef methods[] = {
    { "from_2d", pycudamat_from_2d, METH_VARARGS, "Generate a matrix from a 2d list structure."},
    { "zeros", pycudamat_zeros, METH_VARARGS, "Generate a 0.0-initialized matrix." },
	{ "test", pycudamat_test, METH_VARARGS, "Test function."},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef pycudamat_module = {
    PyModuleDef_HEAD_INIT,
    "pycudamat",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit_pycudamat(void)
{
    PyObject* module;

    if (PyType_Ready(&MatrixType) < 0) {
        return NULL;
    }

    module = PyModule_Create(&pycudamat_module);

    if (NULL == module) {
        return NULL;
    }

    Py_INCREF(&MatrixType);
    PyModule_AddObject(module, "Matrix", (PyObject*)&MatrixType);

    return (PyObject*) module;
}

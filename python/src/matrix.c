#include <Python.h>

#include "..\..\cuda\include\matrix.cuh"

typedef struct {
    PyObject_HEAD

    matrix_t matrix;
} MatrixObject;

static void Matrix_dealloc(MatrixObject* self);
static PyObject* Matrix_new(PyTypeObject* type, PyObject* args, PyObject *kwds);
static int Matrix_init(MatrixObject* self, PyObject* args, PyObject *kwds);
static PyObject* Matrix_repr(PyObject* self);
static PyObject* Matrix_str(PyObject* self);

static PyObject* Matrix_cell(MatrixObject* self, PyObject* args, PyObject *kwds);
static PyObject* Matrix_shape(MatrixObject* self, PyObject* args, PyObject *kwds);
static PyObject* Matrix_mult(MatrixObject* self, PyObject* args, PyObject* kwds);

static PyMethodDef Matrix_methods[] = {
    { "cell", (PyCFunction) Matrix_cell, METH_VARARGS, "Returns cell specified by row and column." },
    { "shape", (PyCFunction) Matrix_shape, METH_NOARGS, "Retunrs shape of the matrix." },
    { "mult", (PyCFunction) Matrix_mult, METH_VARARGS, "Computes matrix product AB of matrices A and B." },
    { NULL }
};

static PyTypeObject MatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "cudamat.Matrix",
    .tp_doc = "Matrix object",
    .tp_basicsize = sizeof(MatrixObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Matrix_new,
    .tp_init = (initproc) Matrix_init,
    .tp_dealloc = (destructor) Matrix_dealloc,
    .tp_methods = Matrix_methods,
    .tp_repr = Matrix_repr,
    .tp_str = Matrix_str
};

static void Matrix_dealloc(MatrixObject* self)
{
    destruct(self->matrix);

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Matrix_new(PyTypeObject* type, PyObject* args, PyObject *kwds)
{
    MatrixObject* self = (MatrixObject*)type->tp_alloc(type, 0);

    return (PyObject*) self;
}

static int Matrix_init(MatrixObject* self, PyObject* args, PyObject *kwds)
{
    return 0;
}

static PyObject* Matrix_repr(PyObject* self)
{
    MatrixObject* mat = (MatrixObject*) self;

    size_t rows = mat->matrix.rows;
    size_t cols = mat->matrix.cols;

    size_t repr_size = 9 + (rows - 1) * (1 + ((cols-1) * 64) + 64 + 3) + 1 + ((cols-1) * 64 + 64 + 2) + 1;
    char* repr = malloc(sizeof(char) * repr_size);

    strcat(repr, "Matrix=[\n");

    for (size_t i = 0; i < rows; ++i) {
        strcat(repr, "[ ");

        for (size_t j = 0; j < cols; ++j) {
            char tmp[64];

            double* val = NULL;

            if (cell(mat->matrix, i, j, &val) == OUT_OF_BOUNDS) {
                return NULL;
            }

            sprintf(tmp, "%.3f ", *val);

            strcat(repr, tmp);
        }

        strcat(repr, "]\n");
    }

    strcat(repr, "]");

    PyObject* result = PyUnicode_FromString(repr);

    free(repr);

    return result;
}

static PyObject* Matrix_str(PyObject* self)
{
    return NULL;
}

static PyObject* Matrix_cell(MatrixObject* self, PyObject* args, PyObject* kwds)
{
    unsigned int row, col;

    if (!PyArg_ParseTuple(args, "II", &row, &col)) {
        return NULL;
    }

    double* value = NULL;

    if (cell(self->matrix, row, col, &value) == OUT_OF_BOUNDS) {
        PyErr_SetString(PyExc_IndexError, "Index out of bounds!");
        
        return NULL;
    }

    return PyFloat_FromDouble(*value);
}

static PyObject* Matrix_shape(MatrixObject* self, PyObject* args, PyObject* kwds)
{
    return Py_BuildValue("(II)", self->matrix.rows, self->matrix.cols);
}

static PyObject* Matrix_mult(MatrixObject* self, PyObject* args, PyObject* kwds)
{
    MatrixObject* A = self;
    MatrixObject* B;

    if (!PyArg_ParseTuple(args, "O!", &MatrixType, &B)) {
        return NULL;
    }

    size_t m, n;

    m = A->matrix.rows;
    n = B->matrix.cols;

    MatrixObject* C = (MatrixObject*) PyObject_CallObject((PyObject*) &MatrixType, NULL);

    C->matrix = zeros(m, n);

    mult(A->matrix, B->matrix, &(C->matrix));

    return (PyObject*) C;
}

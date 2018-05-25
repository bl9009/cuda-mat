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
static PyObject* Matrix_richcompare(MatrixObject* self, MatrixObject* other, int op);

static PyObject* Matrix_getitem(MatrixObject* self, Py_ssize_t i);
static int Matrix_setitem(MatrixObject* self, Py_ssize_t i, PyObject* value);

static PyObject* Matrix_cell(MatrixObject* self, PyObject* args, PyObject *kwds);
static PyObject* Matrix_shape(MatrixObject* self, PyObject* args, PyObject *kwds);
static PyObject* Matrix_mult(MatrixObject* self, PyObject* args, PyObject* kwds);
static PyObject* Matrix_transpose(MatrixObject* self, PyObject* args, PyObject* kwds);

static PyMethodDef Matrix_methods[] = {
    { "cell", (PyCFunction) Matrix_cell, METH_VARARGS, "Returns cell specified by row and column." },
    { "shape", (PyCFunction) Matrix_shape, METH_NOARGS, "Retunrs shape of the matrix." },
    { "mult", (PyCFunction) Matrix_mult, METH_VARARGS, "Computes matrix product AB of matrices A and B." },
    { "transpose", (PyCFunction) Matrix_transpose, METH_NOARGS, "Computes transposition of matrix A." },
    { NULL }
};

static PySequenceMethods seq_methods = {
    .sq_item = (ssizeargfunc) Matrix_getitem,
    .sq_ass_item = (ssizeobjargproc) Matrix_setitem
};

static PyTypeObject MatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pycudamat.Matrix",
    .tp_doc = "Matrix object",
    .tp_basicsize = sizeof(MatrixObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Matrix_new,
    .tp_init = (initproc) Matrix_init,
    .tp_dealloc = (destructor) Matrix_dealloc,
    .tp_methods = Matrix_methods,
    .tp_repr = Matrix_repr,
    .tp_str = Matrix_str,
    .tp_richcompare = (richcmpfunc) Matrix_richcompare,
    .tp_as_sequence = &seq_methods
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

static PyObject* Matrix_richcompare(MatrixObject* self, MatrixObject* other, int op)
{
    PyObject* result = NULL;

    if (!PyObject_TypeCheck(other, &MatrixType)) {
        result = Py_NotImplemented;
    }
    else {
        switch (op) {
            case Py_EQ:
            case Py_NE:
                matrix_t A = self->matrix;
                matrix_t B = other->matrix;

                result = (op == Py_EQ) ? Py_True : Py_False;

                if (A.rows != B.rows || A.cols != B.cols) {
                    result = (op == Py_EQ) ? Py_False : Py_True;
                }

                for (int i = 0; i < A.rows; ++i) {
                    for (int j = 0; j < A.cols; ++j) {
                        double* val_A;
                        double* val_B;

                        cell(A, i, j, &val_A);
                        cell(B, i, j, &val_B);

                        if (*val_A != *val_B) {
                            result = (op == Py_EQ) ? Py_False : Py_True;
                        }
                    }
                }

                break;

            default:
                result = Py_NotImplemented;
        }
    }

    Py_XINCREF(result);

    return result;
}

static PyObject* Matrix_getitem(MatrixObject* self, Py_ssize_t i)
{
    if (self->matrix.rows > 1) {
        if ((size_t)i >= self->matrix.rows) {
            PyErr_SetString(PyExc_IndexError, "Index out of bounds!");

            return NULL;
        }

        MatrixObject* row = (MatrixObject*) PyObject_CallObject((PyObject*) &MatrixType, NULL);

        size_t n = self->matrix.cols;

        row->matrix = zeros(1, n);

        for (int j = 0; j < n; ++j) {
            double* src;
            double* dst;

            cell(self->matrix, i, j, &src);
            cell(row->matrix, 0, j, &dst);

            *dst = *src;
        }

        return (PyObject*) row;
    }
    else {
        if ((size_t)i >= self->matrix.cols) {
            PyErr_SetString(PyExc_IndexError, "Index out of bounds!");

            return NULL;
        }

        double* res;

        cell(self->matrix, 0, i, &res);

        return PyFloat_FromDouble(*res);
    }

}

static int Matrix_setitem(MatrixObject* self, Py_ssize_t i, PyObject* value)
{
    return 0;
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

    if (mult(A->matrix, B->matrix, &(C->matrix)) == SHAPE_MISMATCH) {
        PyErr_SetString(PyExc_ValueError, "Shapes of matrices do not match!");

        return NULL;
    }

    return (PyObject*) C;
}

static PyObject* Matrix_transpose(MatrixObject* self, PyObject* args, PyObject* kwds)
{
    size_t m, n;

    m = self->matrix.cols;
    n = self->matrix.rows;

    MatrixObject* A_t = (MatrixObject*) PyObject_CallObject((PyObject*) &MatrixType, NULL);

    A_t->matrix = zeros(m, n);

    transpose(self->matrix, &A_t->matrix);

    return (PyObject*) A_t;
}

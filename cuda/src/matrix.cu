#include <stdio.h>
#include <stdlib.h>

#include "..\include\matrix.cuh"

typedef matrix_t h_matrix_t; // host matrix type
typedef matrix_t d_matrix_t; // device matrix type
typedef vector_t h_vector_t;
typedef vector_t d_vector_t;

#ifdef __cplusplus
extern "C" {
#endif

// private interface
d_matrix_t load_device(matrix_t A);
void unload_device(d_matrix_t d_mat);
void fetch_unload_device(d_matrix_t d_mat, h_matrix_t* h_mat);

#ifdef __cplusplus
}
#endif

// public interface
static matrix_t construct(size_t rows, size_t cols)
{
    matrix_t mat = { 0, 0, rows, cols };

    mat.pitch = cols * sizeof(double);
    mat.data = (double*)malloc(mat.pitch * rows);

    return mat;
}

matrix_t zeros(size_t rows, size_t cols)
{
    matrix_t A = construct(rows, cols);

    for (int i = 0; i < rows * cols; ++i) {
        A.data[i] = 0.0;
    }

    return A;
}

__device__ double* cell_device(d_matrix_t A, size_t row, size_t col)
{
    double* r = (double*)((char*)A.data + row * A.pitch);
    return &r[col];
}

error_t cell(matrix_t A, size_t row, size_t col, double** value)
{
    if (row >= A.rows || col >= A.cols) {
        return OUT_OF_BOUNDS;
    }

    double* r = (double*)((char*) A.data + row * A.pitch);
    *value = &r[col];

    return OK;
}

error_t from_seq(double* seq, matrix_t* A)
{
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            double* tmp = NULL;

            if (cell(*A, i, j, &tmp) == OUT_OF_BOUNDS) {
                return OUT_OF_BOUNDS;
            }

            *tmp = seq[i * A->cols + j];
            //*cell(*A, i, j) = seq[i * A->cols + j];
        }
    }

    return OK;
}

shape_t mult_result_shape(matrix_t A, matrix_t B)
{
    shape_t shape = { A.rows, B.cols };

    return shape;
}

__global__ void mult_kernel(d_matrix_t d_A, d_matrix_t d_B, d_matrix_t d_C)
{
    size_t row = threadIdx.x;
    size_t col = threadIdx.y;

    double* c = cell_device(d_C, row, col);

    double sum = 0.0;

    for (int k = 0; k < d_A.cols; ++k) {
        sum += *cell_device(d_A, row, k) * *cell_device(d_B, k, col);
    }

    *c = sum;

}

error_t mult(matrix_t A, matrix_t B, matrix_t* C)
{
    h_matrix_t h_C = *C;

    size_t res_rows = A.rows;
    size_t res_cols = B.cols;

    if (res_rows != res_cols ||
        res_rows != h_C.rows ||
        res_cols != h_C.cols) {
        return SHAPE_MISMATCH;
        }

    d_matrix_t d_A = load_device(A);
    d_matrix_t d_B = load_device(B);
    d_matrix_t d_C = load_device(h_C);

    mult_kernel<<<dim3(1,1,1), dim3((unsigned)d_C.rows, (unsigned)d_C.cols,1)>>>(d_A, d_B, d_C);

    cudaThreadSynchronize();

    unload_device(d_A);
    unload_device(d_B);

	fetch_unload_device(d_C, &h_C);

    return OK;
}

__global__ void transpose_kernel(d_matrix_t d_A, d_matrix_t d_A_t)
{
    size_t row = threadIdx.x;
    size_t col = threadIdx.y;

    *cell_device(d_A_t, row, col) = *cell_device(d_A, col, row);
}

shape_t transpose_result_shape(matrix_t A)
{
    shape_t shape = { A.cols, A.rows };

    return shape;
}

error_t transpose(matrix_t A, matrix_t* A_t)
{
    h_matrix_t h_A = A;
    h_matrix_t h_A_t = *A_t;

    d_matrix_t d_A = load_device(h_A);
    d_matrix_t d_A_t = load_device(h_A_t);

    transpose_kernel<<<dim3(1,1,1),dim3((unsigned)d_A_t.rows, (unsigned)d_A_t.cols, 1)>>>(d_A, d_A_t);

    unload_device(d_A);
	fetch_unload_device(d_A_t, &h_A_t);

    return OK;
}

void destruct(matrix_t A)
{
	free(A.data);
}

void print(matrix_t mat)
{
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            double* tmp = NULL;

            if (cell(mat, i, j, &tmp) == OUT_OF_BOUNDS) {
                printf("index out of bounds!");

                return;
            }

            printf("[%f] ", *tmp);
        }

        printf("\n");
    }

    printf("\n");
}

d_matrix_t load_device(h_matrix_t h_mat)
{
    d_matrix_t d_mat = { 0, 0, h_mat.rows, h_mat.cols };

    cudaMallocPitch(&d_mat.data,
        &d_mat.pitch,
        sizeof(double) * h_mat.cols,
        h_mat.rows);

    cudaMemcpy2D(d_mat.data,
        d_mat.pitch,
        h_mat.data,
        h_mat.pitch,
        sizeof(double) * h_mat.cols,
        h_mat.rows,
        cudaMemcpyHostToDevice);

    return d_mat;
}

void unload_device(d_matrix_t d_mat)
{
    cudaFree(d_mat.data);
}

void fetch_unload_device(d_matrix_t d_mat, h_matrix_t* h_mat)
{
    cudaMemcpy2D((*h_mat).data,
        (*h_mat).pitch,
        d_mat.data,
        d_mat.pitch,
        sizeof(double) * d_mat.cols,
        d_mat.rows,
        cudaMemcpyDeviceToHost);

    cudaFree(d_mat.data);
}

#define construct NOT_ALLOWED
#define load_device NOT_ALLOWED
#define unload_device NOT_ALLOWED
#define mult_kernel NOT_ALLOWED
#define transpose_kernel NOT_ALLOWED

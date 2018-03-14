#include <stdio.h>
#include <stdlib.h>

struct {
    double** data;
    unsigned m;
    unsigned n;
} typedef matrix_t;

double** alloc_grid(unsigned m)
{
#ifdef __CUDACC__
    double** tmp;
    cudaMallocManaged(&tmp, m * sizeof(double*));

    return tmp;
#else
    return (double**)malloc(m * sizeof(double*));
#endif
}

double* alloc_row(unsigned n)
{
#ifdef __CUDACC__
    double* tmp;
    cudaMallocManaged(&tmp, n * sizeof(double));

    return tmp;
#else
    return (double*)malloc(n * sizeof(double));
#endif
}

matrix_t zeros(unsigned m, unsigned n)
{
    matrix_t mat = { alloc_grid(m), m, n };

    for (unsigned i = 0; i < m; ++i) {
        mat.data[i] = alloc_row(n);

        for (unsigned j = 0; j < n; ++j) {
            mat.data[i][j] = 0.0;
        }
    }

    return mat;
}

matrix_t from_seq(double* seq, unsigned m, unsigned n)
{
    matrix_t mat = { alloc_grid(m), m, n };

    for (unsigned i = 0; i < m; ++i) {
        mat.data[i] = alloc_row(n);

        for (unsigned j = 0; j < n; ++j) {
            mat.data[i][j] = *seq;
            ++seq;
        }
    }

    return mat;
}

#ifdef __CUDACC__
__global__
#endif
void compute_cell(matrix_t A, matrix_t B, matrix_t C, unsigned i, unsigned j)
{
    double sum = 0.0;

    for (int k = 0; k < A.n; ++k) {
        sum += A.data[i][k] * B.data[k][j];
    }

    C.data[i][j] = sum;
}

matrix_t mult(matrix_t A, matrix_t B)
{
    if (A.n != B.m) {
        // workaround for MSVC....
        matrix_t mat = { 0, 0, 0 };
        return mat;
        
        //return ((matrix_t) { 0, 0, 0 });
    }

    matrix_t C = zeros(A.m, B.n);

    for (unsigned i = 0; i < C.m; ++i) {
        for (unsigned j = 0; j < C.n; ++j) {
#ifdef __CUDACC__
            compute_cell<<<1,1>>>(A, B, C, i, j);
#else
            compute_cell(A, B, C, i, j);
#endif
        }
    }

#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif

    return C;
}

#ifdef __CUDACC__
__device__
void find_result_pos(unsigned thread_id, unsigned m, unsigned n, unsigned& i, unsigned& j)
{
    j = thread_id % n;
    i = (thread_id - j) / n;
}

__global__
void mult2_kernel(matrix_t A, matrix_t B, matrix_t& C, int& result)
{
    if (A.n != B.m) {
        result = -1;

        return;
    }

    unsigned i, j;
    find_result_pos(threadIdx.x, C.m, C.n, i, j);

    double sum = 0.0;

    for (int k = 0; k < A.n; ++k) {
        sum += A.data[i][k] * B.data[k][j];
    }

    C.data[i][j] = sum;

    result = 0;
}

matrix_t mult2(matrix_t A, matrix_t B)
{
    if (A.n != B.m) {
        matrix_t tmp = { 0, 0, 0 };
        return tmp;
    }

    matrix_t C = zeros(A.n, B.m);

    int result;
    mult2_kernel<<<1, 1>>>(A, B, C, result);

    cudaDeviceSynchronize();

    if (result != 0) {
        matrix_t tmp = { 0, 0, 0 };
        return tmp;
    }
    else {
        return C;
    }
}
#endif

void clear(matrix_t mat)
{
    for (unsigned i = 0; i < mat.m; ++i) {
#ifdef __CUDACC__
        cudaFree(mat.data[i]);
#else
        free(mat.data[i]);
#endif
    }

#ifdef __CUDACC__
    cudaFree(mat.data);
#else
    free(mat.data);
#endif
}

void print(matrix_t mat)
{
    for (unsigned i = 0; i < mat.m; ++i) {
        for (unsigned j = 0; j < mat.n; ++j) {
            printf("[%f] ", mat.data[i][j]);
        }

        printf("\n");
    }
}


/*
namespace cudamat
{
    typedef std::vector<double> MatrixRow;
    typedef std::vector<MatrixRow> MatrixData;
}

namespace cudamat
{
    struct MatrixShape
    {
        size_t rows;
        size_t cols;
    };
}

namespace cudamat
{
    class Matrix;

    __global__
    void computeCell(const Matrix& A, const Matrix& B, size_t i, size_t j, size_t m, Matrix& C);
}

namespace cudamat
{
    class Matrix
    {
        public:
            static Matrix create(const MatrixData& data)
            {
                if (Matrix::shapeValid(data))
                {
                    return Matrix(data);
                }
                else
                {
                    throw std::exception();
                }
            }

            static Matrix zeros(size_t rows, size_t cols)
            {
                MatrixData data;

                for (size_t i = 0; i < rows; ++i)
                {
                    MatrixRow row;

                    for (size_t j = 0; j < cols; ++j)
                    {
                        row.push_back(0.0);
                    }

                    data.push_back(row);
                }

                return Matrix(data);
            }

            static Matrix zeros(const MatrixShape& shape)
            {
                return Matrix::zeros(shape.rows, shape.cols);
            }

            static void print(const Matrix& matrix)
            {
                MatrixData data = matrix.data();

                for (const auto& row : data)
                {
                    for (const auto& cell : row)
                    {
                        std::cout << "[" << cell << "] ";
                    }

                    std::cout << std::endl;
                }
            }

        public:
            Matrix mult(const Matrix& B) const
            {
                const Matrix A = *this;

                const size_t m = A.shape().cols;
                const size_t n = A.shape().rows;
                const size_t p = B.shape().cols;
                const size_t q = B.shape().rows;

                if (m != q)
                {
                    throw std::exception();
                }

                Matrix C = Matrix::zeros(n, p);

                for (size_t i = 0; i < n; ++i)
                {
                    for (size_t j = 0; j < p; ++j)
                    {
                        // double sum = 0;

                        // for (int k = 0; k < m; ++k)
                        // {
                        //     sum += A[i][k] * B[k][j];
                        // }

                        // C[i][j] = sum;

                        //C[i][j] = computeCell(A, B, i, j, m);
                        computeCell<<<1,1>>>(A, B, i, j, m, C);
                    }
                }

                return C;
            }

            MatrixData data() const { return _data; }
            MatrixShape shape() const { return _shape; }

            MatrixRow operator [](size_t i) const { return _data[i]; }
            MatrixRow& operator [](size_t i) { return _data[i]; }

        private:
            Matrix(const MatrixData& data)
                : _data(data)
            {
                _shape = evalShape(data);
            }

        private:
            static bool shapeValid(const MatrixData& data)
            {
                if (data.empty() || data[0].empty())
                {
                    return false;
                }

                size_t refShape = data[0].size();

                for (const auto& cells : data)
                {
                    if (cells.size() != refShape)
                    {
                        return false;
                    }
                }

                return true;
            }

            MatrixShape evalShape(const MatrixData& data)
            {
                MatrixShape shape;

                shape.rows = data.size();
                shape.cols = data[0].size();

                return shape;
            }

        private:
            MatrixData _data;

            MatrixShape _shape;
    };
}

namespace cudamat
{
    __global__
    void computeCell(const Matrix& A, const Matrix& B, size_t i, size_t j, size_t m, Matrix& C)
    {
        double result = 0.0;

        for (size_t k = 0; k < m; ++k)
        {
            result += A[i][k] * B[k][j];
        }

        C[i][j] = result;
    }
}


*/

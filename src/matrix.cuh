#include <cstddef>
#include <iostream>

#include <exception>
#include <vector>

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

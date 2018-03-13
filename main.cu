#include <stdio.h>

double** zeros(unsigned m, unsigned n)
{
    double** matrix = (double**)malloc(m * sizeof(double*));

    for (unsigned i = 0; i < m; ++i)
    {
        matrix[i] = (double*)malloc(n * sizeof(double));

        for (unsigned j = 0; j < n; ++j)
        {
            matrix[i][j] = 0.0;
        }
    }

    return matrix;
}

double** mult(double** A, unsigned m, unsigned n, double** B, unsigned p, unsigned q)
{
    if (n != q) {
        return 0;
    }

    double** C = zeros(m, p);

    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < p; ++j)
        {
            double sum = 0.0;

            for (int k = 0; k < m; ++k)
            {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = sum;

            //C[i][j] = computeCell(A, B, i, j, m);
            //computeCell << <1, 1 >> >(A, B, i, j, m, C);
        }
    }

    return C;
}

void print(double** matrix, unsigned m, unsigned n)
{
    for (unsigned i = 0; i < m; ++i)
    {
        for (unsigned j = 0; j < n; ++j)
        {
            printf("[%f] ", matrix[i][j]);
        }

        printf("\n");
    }
}

int main()
{
    //double** A = zeros(4, 5);

    double A[2][3]{ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    double B[3][2]{ { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };

    double** C = mult(&A, 2, 3, B, 3, 2);

    print(A, 4, 5);

    return 0;
}


/*#include <iostream>

#include "src/matrix.cuh"

int main()
{
    std::cout << "my mat lib" << std::endl;

    cudamat::MatrixData dataA{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 }, { 13, 14, 15 } };
    cudamat::MatrixData dataB{ { 1, 2, 7 }, { 3, 4, 8 }, { 5, 6, 9 } };

    cudamat::Matrix A = cudamat::Matrix::create(dataA);
    cudamat::Matrix B = cudamat::Matrix::create(dataB);

    std::cout << "A shape - rows: " << A.shape().rows << ", cols: " << A.shape().cols << std::endl;
    std::cout << "B shape - rows: " << B.shape().rows << ", cols: " << B.shape().cols << std::endl;

    cudamat::Matrix C = A.mult(B);

    std::cout << "C shape - rows: " << C.shape().rows << ", cols: " << C.shape().cols << std::endl;

    cudamat::Matrix::print(C);

    return 0;
}*/

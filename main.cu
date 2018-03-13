#include <iostream>

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
}

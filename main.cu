#include <stdio.h>
#include <stdlib.h>

#include "src/matrix.cuh"

int main()
{
    double seqA[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    unsigned m = 2;
    unsigned n = 3;

    double seqB[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    unsigned p = 3;
    unsigned q = 2;

    matrix_t A = from_seq(seqA, m, n);
    matrix_t B = from_seq(seqB, p, q);

    matrix_t C = mult(A, B);

    print(C);

#ifdef __CUDACC__
    matrix_t D = mult2(A, B);
    print(D);
#endif

    clear(A);
    clear(B);
    clear(C);
    printf("here12");
#ifdef __CUDACC__
    clear(D);
    printf("here13");
#endif

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

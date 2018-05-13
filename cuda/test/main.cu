#include <stdio.h>
#include <stdlib.h>

#include "include\matrix.cuh"

int main()
{
    double seq_A[6] = { 1., 2., 3., 4., 5., 6. };
    double seq_B[6] = { 1., 2., 3., 4., 5., 6. };

    matrix_t A = zeros(2, 3);
    matrix_t B = zeros(3, 2);

    from_seq(seq_A, &A);
    from_seq(seq_B, &B);

    shape_t mult_s = mult_result_shape(A, B);

    matrix_t C = zeros(mult_s);

    mult(A, B, &C);

    shape_t trans_s = transpose_result_shape(A);

    matrix_t A_t = zeros(trans_s);

    transpose(A, &A_t);

    // some memory leaks
    //mult(A, B);
    //matrix_t D = transpose(from_seq(seq_A, 2, 3));

    print(A);
    print(B);
    print(C);

    print(A_t);

    destruct(A);
    destruct(B);
    destruct(C);
    destruct(A_t);

    return 0;
}

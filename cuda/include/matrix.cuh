/*
* cuda-mat
* A CUDA LINEAR ALGEBRA library.
*
* Provides an interface for basic 2D matrix and
* vector computations like multiplication, transposal etc.
*
*/

#ifndef MATRIX_CUH_
#define MATRIX_CUH_

#ifdef __cplusplus
extern "C" {
#endif

/*---------------------------------------------------*/
/*                 TYPE DEFINITIONS                  */
/*___________________________________________________*/

/*
 * Describes a matrix of shape rows*cols.
 */
struct {
    double* data;
    size_t pitch;
    size_t rows;
    size_t cols;
} typedef matrix_t;

/*
 * Describes a vector of shape rows*1 or 1*cols if transposed.
 */
typedef matrix_t vector_t;

/*
 * Represents the shape of a matrix.
 */
struct {
    size_t rows;
    size_t cols;
} typedef shape_t;

/*
 * Error value enum returned by library functions.
 *     - OK: Operation returned as expected.
 *     - SHAPE_MISSMATCH: The shapes of involved matrices do not match.
 */
enum { OK, SHAPE_MISMATCH, OUT_OF_BOUNDS } typedef error_t;

/*---------------------------------------------------*/
/*              FUNCTION DECLARATIONS                */
/*___________________________________________________*/

/*
 * Construct and return a zero-initiliazed 2D matrix.
 *
 * @param rows: Number of rows.
 * @param cols: Number of cols.
 *
 * @return matrix_t: The zero-initialized matrix.
 */
matrix_t zeros(size_t rows, size_t cols);

/*
 * Access a cell of a matrix specified by row and column.
 *
 * @param A: The matrix.
 * @param row: Row location of the cell.
 * @param col: Column location of the cell.
 * @param value: Pointer to the value for call-by-reference.
 *
 * @return OUT_OF_BOUNDS if index is out of bounds, OK otherwise.
 */
error_t cell(matrix_t A, size_t row, size_t col, double** value);

/*
 * Loads values from a sequence into a given matrix.
 *
 * @param seq: A sequence of double values that shall be loaded.
 * @param A: Pointer to the matrix that shall be loaded.
 *
 * @return OK if operation executed as expected.
 */
error_t from_seq(double* seq, matrix_t* A);

/*
 * Calculates the shape of the matrix resulting of a multiplication
 * of the given matrices. The result of this operation can be used
 * to construct a result matrix for the multiplication.
 *
 * @param A: The multiplier.
 * @param B: The multiplicand.
 *
 * @return Expected shape of the multiplication.
 */
shape_t mult_result_shape(matrix_t A, matrix_t B);

/*
 * Computes the matrix product AB of two matrices A and B.
 *
 * Note that the resulting matrix C needs to be constructed beforehand
 * using the shape calculated by the mult_result_shape function.
 *
 * @param A: The multiplier.
 * @param B: The multiplicand.
 * @param C: Result of the matrix product AB.
 *
 * @return OK if operation finished as expected, SHAPE_MISMATCH if shape of
 *   both matrices do mismatch.
 */
error_t mult(matrix_t A, matrix_t B, matrix_t* C);

/*
 * Calculates the shape of the transpose matrix of the given matrix.
 * The result of this operation can be used to construct a result
 * matrix for the transposition.
 *
 * @param A: The matrix to transpose.
 *
 * @return Expected shape of the transposition.
 */
shape_t transpose_result_shape(matrix_t A);

/*
 * Calculates the transpose matrix of the given matrix.
 *
 * @param A: The matrix to transpose.
 * @param A_t: Pointer to the result matrix for the transpose matrix.
 *
 * @return OK if operation finished as expected.
 */
error_t transpose(matrix_t A, matrix_t* A_t);

/*
 * Frees memory allocated by a matrix.
 *
 * @param A: The matrix structure to be deallocated and freed.
 */
void destruct(matrix_t A);

/*
 * Prints the content of the given matrix to stdout. For debugging only.
 *
 * @param mat: The matrix to be printed.
 */
void print(matrix_t mat);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_CUH_

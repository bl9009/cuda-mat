import unittest

import pycudamat as pcm

class TestMatrix(unittest.TestCase):

    def test_from_2d(self):
        l = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        m = pcm.from_2d(l)

        self.assertEqual(type(m), pcm.Matrix)
        self.assertEqual(m.shape(), (3, 3))

        for i, row in enumerate(l):
            for j, cell in enumerate(row):
                self.assertEqual(m.cell(i, j), cell)

    def test_zeros(self):
        rows = 10
        cols = 12

        m = pcm.zeros(rows, cols)

        self.assertEqual(type(m), pcm.Matrix)
        self.assertEqual(m.shape(), (rows, cols))

        for i in range(rows):
            for j in range(cols):
                self.assertEqual(m.cell(i, j), 0.0)

    def test_cell(self):
        l = [[1, 2, 3], [4, 5, 6]]

        m = pcm.from_2d(l)

        self.assertEqual(m.cell(1, 1), float(l[1][1]))
        self.assertRaises(IndexError, m.cell, 3, 0)

    def test_set(self):
        l = [[1, 2, 3], [4, 5, 6]]

        m = pcm.from_2d(l)

        m.set(1, 1, 42)

        self.assertEqual(m.cell(1, 1), 42)

    def test_get_item(self):
        l = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        m = pcm.from_2d(l)

        for i, row in enumerate(l):
            for j, cell in enumerate(row):
                self.assertEqual(m[i][j], l[i][j])
                self.assertEqual(m[i][j], cell(i, j))

    def test_set_item(self):
        l = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        m = pcm.from_2d(l)

        m[1][2] = 42

        self.assertEqual(m.cell(1, 2), 42)

    def test_shape(self):
        rows = 10
        cols = 12

        m = pcm.zeros(rows, cols)

        self.assertEqual(m.shape(), (rows, cols))

    def test_mult(self):
        A = pcm.from_2d([[1, 2, 3], [4, 5, 6]])
        B = pcm.from_2d([[1, 2], [3, 4], [5, 6]])
        E = pcm.from_2d([[0]])

        C = pcm.from_2d([[22, 28], [49, 64]])
        C_test = A.mult(B)

        self.assertEqual(C.shape(), C_test.shape())

        for i in range(2):
            for j in range(2):
                self.assertEqual(C.cell(i, j), C_test.cell(i, j))

        self.assertRaises(ValueError, A.mult, E)

    def test_transpose(self):
        A = pcm.from_2d([[1, 2, 3], [4, 5, 6]])
        A_t = pcm.from_2d([[1, 4], [2, 5], [3, 6]])

        A_t_test = A.transpose()

        for i in range(3):
            for j in range(2):
                self.assertEqual(A_t_test.shape(), A_t.shape())
                self.assertEqual(A_t_test.cell(i, j), A_t.cell(i, j))
#if __name__ == '__main__':
#    test_init()

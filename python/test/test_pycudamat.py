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

    def test_shape(self):
        rows = 10
        cols = 12

        m = pcm.zeros(rows, cols)

        self.assertEqual(m.shape(), (rows, cols))

    def test_mult(self):
        A = pcm.from_2d([[1, 2, 3], [4, 5, 6]])
        B = pcm.from_2d([[1, 2], [3, 4], [5, 6]])

        C = pcm.from_2d([[22, 28], [49, 64]])
        C_test = A.mult(B)

        self.assertEqual(C.shape(), C_test.shape())

        for i in range(2):
            for j in range(2):
                self.assertEqual(C.cell(i, j), C_test.cell(i, j))

#if __name__ == '__main__':
#    test_init()
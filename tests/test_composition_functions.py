import numpy as np
import torch
from scripts.composition_functions import tw_transform, transweigh_weight, concat
import unittest


class CompositionFunctionsTest(unittest.TestCase):
    """
    This class tests the composition functions.
    This test suite can be ran with:
        python -m unittest -q tests.CompositionFunctionsTest
    """

    def setUp(self):
        self.u = torch.from_numpy(np.array([[1, 1, 1]], dtype='float32'))
        self.v = torch.from_numpy(np.array([[1, 0, 0]], dtype='float32'))

        self.transformations_tensor = np.full(shape=(2, 6, 3), fill_value=0.0, dtype='float32')
        two_eyes = np.concatenate((np.eye(3), np.eye(3)), axis=0)
        self.transformations_tensor[0] = np.copy(two_eyes)
        self.transformations_tensor[1] = np.copy(two_eyes)
        self.transformations_tensor = self.set_tensor_values(tensor=self.transformations_tensor, matrix=[0, 1, 1, 1],
                                                             column=[2, 0, 1, 3], row=[2, 0, 1, 0], values=[0] * 4)
        self.transformations_tensor = torch.from_numpy(self.transformations_tensor)

        self.transformations_bias = torch.from_numpy(np.full(shape=(2, 3), fill_value=0.0, dtype='float32'))

        self.combining_tensor = np.full(shape=(2, 3, 3), fill_value=0.0, dtype='float32')
        matrix_numbers = [0] * 9 + [1] * 9
        column_numbers = [0] * 3 + [1] * 3 + [2] * 3 + [0] * 3 + [1] * 3 + [2] * 3
        row_numbers = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        values = [1, 1, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 1, 0, 1, 2, 2]
        self.combining_tensor = self.set_tensor_values(tensor=self.combining_tensor, matrix=matrix_numbers,
                                                       column=column_numbers,
                                                       row=row_numbers, values=values)
        self.combining_tensor = torch.from_numpy(self.combining_tensor)

        self.combining_bias = torch.from_numpy(np.full(shape=(3,), fill_value=0.0, dtype='float32'))

    @staticmethod
    def set_tensor_values(tensor, matrix, column, row, values):
        """
        Helper method to initialize a third-order tensor with specific values
        """
        for i in range(len(matrix)):
            tensor[matrix[i]][column[i]][row[i]] = values[i]
        return tensor

    def test_transformation(self):
        """
        Tests that the t transformations are correctly performed.
        transformations_tensor contains t transformations matrices of size 2nxn, where
        n (= embedding_dim) is the size of the input vectors u and v
        """

        # the two transformation matrices are different, so the resulting transformations are also expected to be
        # different, even if they start off with the same inputs u and v
        expected_t = torch.from_numpy(np.array([[[2, 1, 0], [0, 0, 1]]], dtype='float32'))

        t = tw_transform(word1=self.u, word2=self.v, transformation_tensor=self.transformations_tensor,
                         transformation_bias=self.transformations_bias)

        np.testing.assert_allclose(t, expected_t)

    def test_weighting(self):
        """
        Test that the weighting is correctly performed
        If [[A B C] are two transformed representations and [[[g h i] are the transformations weighting matrices
            [D E F]]                                          [j k l]
                                                              [m n p]]
                                                             [[q r s]
                                                              [t u v]
                                                              [x y z]]]
        than the elements of the composed representation [p_0 p_1 p_2] are obtained as:
        p_0 = A*g + B*j + C*m + D*q + E*t + F*x
        p_1 = A*h + B*k + C*n + D*r + E*u + F*y
        p_2 = A*i + B*l + C*p + D*s + E*v + F*z
        """

        t = torch.from_numpy(np.array([[[2, 1, 0], [0, 0, 1]]], dtype='float32'))
        # batch_size = 1
        # no_transformations = 2
        # embedding_dim = 3

        expected_p = torch.from_numpy(np.array([[5, 6, 6]]))

        p = transweigh_weight(trans_uv=t, combining_tensor=self.combining_tensor, combining_bias=self.combining_bias)
        np.testing.assert_allclose(p, expected_p)

    def test_concat(self):
        """
        Test whether two batches of size [1x3] and [1x3] can be concatenated to retrieve a batch of [1x6]
        """
        expected_p = np.array([[1, 1, 1, 1, 0, 0]])
        p = concat(self.u, self.v, axis=1)
        np.testing.assert_allclose(p, expected_p)

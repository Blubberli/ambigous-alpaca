import numpy as np
import torch
from scripts import BasicTwoWordClassifier
import unittest


class BasicTwoWordClassifierTest(unittest.TestCase):
    """
    this class tests the BasicTwoWordClassifier
    This test suite can be ran with:
        python -m unittest -q tests.BasicTwoWordClassifierTest
    """

    def setUp(self):
        self.word1 = torch.from_numpy(np.array([[1, 0, 0]], dtype='float32'))
        self.word2 = torch.from_numpy(np.array([[0, 1, 0]], dtype='float32'))
        self.input_dim = 6
        self.hidden_dim = 6
        self.labels = 2

    def test_forward(self):
        """
        tests the classifier implemented in BasicTwoWordClassifier and the overridden method "forward"
        checks whether the output layer is of the right size
        """
        expected_size = torch.tensor(np.array([[0, 1]])).shape
        classifier = BasicTwoWordClassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, label_nr=self.labels,
                                            dropout_rate=0.0)
        res = classifier.forward(self.word1, self.word2)
        np.testing.assert_allclose(res.shape, expected_size)

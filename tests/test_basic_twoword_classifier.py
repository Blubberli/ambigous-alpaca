import numpy as np
import torch
from scripts import BasicTwoWordClassfier
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
        self.inputdim = 6
        self.hiddendim = 6
        self.labels = 2

    def test_forward(self):
        """
        tests the classifier implemented in BasicTwoWordClassifier and the overridden method "forward"
        prints the classifier to check whether the right number of layers exist and the right dimensions are included
        """
        classifier = BasicTwoWordClassfier(input_dim=self.inputdim, hidden_dim=self.hiddendim, label_nr=self.labels)
        classifier.forward(self.word1, self.word2)
        print(classifier)

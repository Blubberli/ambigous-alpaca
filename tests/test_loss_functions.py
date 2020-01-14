import numpy as np
import torch
from scripts import multi_class_cross_entropy, binary_class_cross_entropy
import unittest

class LossFunctionsTest(unittest.TestCase):
    """
    this class tests the BasicTwoWordClassifier
    This test suite can be ran with:
        python -m unittest -q tests.BasicTwoWordClassifierTest
    """

    def setUp(self):
        self.output_tensor = torch.from_numpy(np.array([[5.0, 5.0, 1.0], [0.0, 1.0, 2.0], [1.0,2.0,3.0]],dtype='float32'))
        self.target_labels = torch.from_numpy(np.array([0,1,2], dtype='long'))

        self.manual_loss = self.calculate_manually(self.output_tensor, self.target_labels)

    @staticmethod
    def calculate_manually(logits, target):
        """
        Helper method to calculate mean loss manually with the help of numpy
        help from: https://d2l.ai/chapter_linear-networks/softmax-regression-scratch.html and http://cs231n.github.io/neural-networks-case-study/#grad
        """
        logits = logits.numpy()
        target = target.numpy()
        num_examples = logits.shape[0]

        exps = [np.exp(i) for i in logits]
        probs = np.array(exps / np.sum(exps, axis=1, keepdims=True)) # softmax

        correct_logprobs = - np.log(probs[range(num_examples), target]) # negative log likelihood

        return np.sum(correct_logprobs)/num_examples



    def test_multiclass_cross_entropy(self):
        """
        tests whether loss from function corresponds to loss manually calculated
        """

        loss = multi_class_cross_entropy(self.output_tensor, self.target_labels)
        np.testing.assert_allclose(loss, self.manual_loss)
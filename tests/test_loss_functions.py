import numpy as np
import torch
from utils.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy
import unittest


class LossFunctionsTest(unittest.TestCase):
    """
    this class tests the BasicTwoWordClassifier
    This test suite can be ran with:
        python -m unittest -q tests.BasicTwoWordClassifierTest
    """

    def setUp(self):
        self.output_tensor = torch.from_numpy(
            np.array([[5.0, 5.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]))
        self.target_labels = torch.from_numpy(np.array([0, 1, 2]))
        self.output_tensor_binary = torch.from_numpy(np.array([0.0, -1.4, -.8, .2, .4, .8, 1.2, 2.2, 2.9, 4.6]))
        self.target_labels_binary = torch.from_numpy(np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


    @staticmethod
    def calculate_manually(logits, target):
        """
        Helper method to calculate mean loss manually with the help of numpy (for multiclass loss)
        help from: https://d2l.ai/chapter_linear-networks/softmax-regression-scratch.html and http://cs231n.github.io/neural-networks-case-study/#grad
        """
        logits = logits.numpy()
        target = target.numpy()
        num_examples = logits.shape[0]

        exps = [np.exp(i) for i in logits]
        probs = np.array(exps / np.sum(exps, axis=1, keepdims=True))  # softmax


        correct_logprobs = - np.log(probs[range(num_examples), target])  # negative log likelihood

        return np.sum(correct_logprobs) / num_examples

    @staticmethod
    def sigmoid(x):
        """
        helper method to calculate sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def cross_entropy(y_hat, y):
        """
        helper method to calculate cross entropy
        """
        if y == 1:
            return -np.log(y_hat)
        else:
            return -np.log(1 - y_hat)

    def test_multiclass_cross_entropy(self):
        """
        multiclass classification: tests whether loss from implemented function corresponds to loss manually calculated
        """
        expected_loss = self.calculate_manually(self.output_tensor, self.target_labels)
        loss = multi_class_cross_entropy(output=self.output_tensor, target=self.target_labels)
        np.testing.assert_allclose(loss, expected_loss)

    def test_binary_cross_entropy(self):
        """
         binary classification: tests whether loss from implemented function corresponds to loss manually calculated
         """
        loss = binary_class_cross_entropy(output=self.output_tensor_binary, target=self.target_labels_binary)

        loss_calc = []
        for x, y in zip(self.output_tensor_binary, self.target_labels_binary): # apply cross entropy and sigmoid to each element in vector
            loss_calc.append(self.cross_entropy(self.sigmoid(x), y).numpy())

        expected_loss = np.mean(loss_calc)

        np.testing.assert_allclose(loss, expected_loss)



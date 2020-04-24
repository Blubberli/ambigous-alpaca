import numpy as np
import math
import torch
from torch import optim
import unittest
from classification_models import TransweighTwoWordClassifier
from utils.loss_functions import binary_class_cross_entropy, multi_class_cross_entropy


class TransweighTwoWordClassifierTest(unittest.TestCase):
    """
    this class tests the TransweighTwoWordClassifier
    This test suite can be ran with:
        python -m unittest -q tests.TransweighTwoWordClassifierTest
    """

    def setUp(self):
        self.model_binary = TransweighTwoWordClassifier(input_dim=4, hidden_dim=2, label_nr=1, transformations=5,
                                                        dropout_rate=0.1,
                                                        normalize_embeddings=True)

        self.model_multiclass = TransweighTwoWordClassifier(input_dim=4, hidden_dim=2, label_nr=4, transformations=5,
                                                            dropout_rate=0.1,
                                                            normalize_embeddings=True)

        self.optimizer_binary = optim.Adam(self.model_binary.parameters(), lr=0.01)
        self.optimizer_multi = optim.Adam(self.model_multiclass.parameters(), lr=0.01)
        self.w1 = torch.from_numpy(np.array([[0.9, 0.5, 1.5, 1.0], [0.1, 0.5, 0.1, 1.0]], dtype=np.float32))
        self.w2 = torch.from_numpy(np.array([[0.5, 0.3, 0.7, 0.1], [0.1, 0.5, 0.6, 0.1]], dtype=np.float32))
        self.label = torch.from_numpy(np.array([[1.0], [0.0]], dtype=np.float32))
        self.label_multi = torch.from_numpy(np.array([3, 1]))
        self.batch_bin = {"w1": self.w1, "w2": self.w2, "l": self.label, "device": "cpu"}
        self.batch_multi = {"w1": self.w1, "w2": self.w2, "l": self.label_multi, "device": "cpu"}

    def test_model_binary(self):
        """
        Test whether the binary classifier can be ran and whether the loss can be computed. The loss should be a number
        larger than zero and not NaN
        """
        self.optimizer_binary.zero_grad()
        output = self.model_binary(self.batch_bin)
        loss = binary_class_cross_entropy(output, self.label).item()
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    def test_model_multiclass(self):
        """
        Test whether the multiclass classifier can be ran and whether the loss can be computed. The loss should be
        a number larger than zero and not NaN
        """
        self.optimizer_multi.zero_grad()
        output = self.model_multiclass(self.batch_multi)
        loss = multi_class_cross_entropy(output, self.label_multi).item()
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    @staticmethod
    def acess_named_parameter(model, parameter_name):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == parameter_name:
                    return param.clone()

    def test_trainable_parameters(self):
        """Test whether the parameter that need to be updated in training are trainable parameters in the model"""
        variables = ["_transformation_tensor", "_transformation_bias", "_combining_tensor", "_combining_bias",
                     "_hidden.weight", "_hidden.bias", "_output.weight", "_output.bias"]
        for name, param in self.model_binary.named_parameters():
            if param.requires_grad:
                np.testing.assert_equal(name in variables, True)

    def test_parameter_get_updated(self):
        """Test whether initial weight matrices are being updated during training. These parameters should be different
        after training vs before training."""
        tw_tensor_before_training = self.acess_named_parameter(self.model_binary, "_transformation_tensor")
        combining_tensor_before_training = self.acess_named_parameter(self.model_binary, "_combining_tensor")

        for epoch in range(0, 10):
            self.optimizer_binary.zero_grad()
            output = self.model_binary(self.batch_bin)
            loss = binary_class_cross_entropy(output, self.label)
            loss.backward()
            self.optimizer_binary.step()
            tw_tensor_after_training = self.acess_named_parameter(self.model_binary, "_transformation_tensor")
            combining_tensor_after_training = self.acess_named_parameter(self.model_binary, "_combining_tensor")
        difference_combining_tensor = torch.sum(
            combining_tensor_before_training - combining_tensor_after_training).item()
        difference_tw_tensor = torch.sum(tw_tensor_before_training - tw_tensor_after_training).item()
        np.testing.assert_equal(difference_combining_tensor != 0.0, True)
        np.testing.assert_equal(difference_tw_tensor != 0.0, True)

    def test_output_shape(self):
        """Test whether the output of both classifiers has the correct shape ([batchsize, lablesize]) and whether the
        composed phrase has the correct shape [batchsize, embedding_dim]"""
        expected_shape_binary = np.array([2, 1])
        expected_shape_multi = np.array([2, 4])
        expected_shape_composed_phrase = np.array([2, 4])
        output_binary = self.model_binary(self.batch_bin)
        composed_phrase = self.model_binary.composed_phrase
        output_multi = self.model_multiclass(self.batch_multi)
        np.testing.assert_almost_equal(output_binary.shape, expected_shape_binary)
        np.testing.assert_almost_equal(output_multi.shape, expected_shape_multi)
        np.testing.assert_almost_equal(composed_phrase.shape, expected_shape_composed_phrase)

    def test_embedding_normalization(self):
        """Test whether the composed phrase has been normalized to unit norm"""
        self.model_binary(self.batch_bin)
        composed_phrase = self.model_binary.composed_phrase
        np.testing.assert_almost_equal(np.linalg.norm(composed_phrase[0].data), 1)

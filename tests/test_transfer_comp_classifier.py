import unittest
from scripts import TransferCompClassifier
from torch import optim
import torch
import pathlib
import numpy as np


class TransferCompClassifierTest(unittest.TestCase):
    """
    this class tests the TransferCompClassifier
    This test suite can be ran with:
        python -m unittest -q tests.TransferCompClassifierTest
    """

    def setUp(self):
        self.path_to_saved_model = pathlib.Path(__file__).parent.absolute().joinpath(
            "models/transweighclassifier_2020-02-26-08_58_55")
        self._pretrained_model = torch.load(self.path_to_saved_model)

        self.model = TransferCompClassifier(input_dim=4, hidden_dim=2, label_nr=1,
                                            dropout_rate=0.1,
                                            normalize_embeddings=True,
                                            pretrained_model=self.path_to_saved_model)
        self.variables = ["_transformation_tensor", "_transformation_bias", "_combining_tensor", "_combining_bias",
                          "_hidden.weight", "_hidden.bias", "_output.weight", "_output.bias"]

        self.optimizer_binary = optim.Adam(self.model.parameters(), lr=0.1)
        self.word1 = torch.from_numpy(np.array([[0.9, 0.5, 1.5, 1.0], [0.1, 0.5, 0.1, 1.0]], dtype=np.float32))
        self.word2 = torch.from_numpy(np.array([[0.5, 0.3, 0.7, 0.1], [0.1, 0.5, 0.6, 0.1]], dtype=np.float32))
        self.label = torch.from_numpy(np.array([[1.0], [0.0]], dtype=np.float32))

    @staticmethod
    def access_named_parameter(model, parameter_name):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == parameter_name:
                    return param.clone()

    def test_initialisation_composition(self):
        """
        tests whether the weights and bias for the composition models are initialised correctly
        comp parameters should be the same as in saved model
        at the same time also tests whether the right parameters are trainable, because otherwise they wouldn't
        be selected in the static method
        """

        model_transformation_tensor = self._pretrained_model["_transformation_tensor"]
        model_transformation_bias = self._pretrained_model["_transformation_bias"]
        model_combination_tensor = self._pretrained_model["_combining_tensor"]
        model_combination_bias = self._pretrained_model["_combining_bias"]

        np.testing.assert_equal((torch.sum(self.access_named_parameter(self.model,
                                                                       "_transformation_tensor") -
                                           model_transformation_tensor).item()) == 0.0,
                                True)
        np.testing.assert_equal((torch.sum(
            self.access_named_parameter(self.model, "_transformation_bias") - model_transformation_bias).item()) == 0.0,
                                True)
        np.testing.assert_equal((torch.sum(
            self.access_named_parameter(self.model, "_combining_tensor") - model_combination_tensor).item()) == 0.0,
                                True)
        np.testing.assert_equal((torch.sum(
            self.access_named_parameter(self.model, "_combining_bias") - model_combination_bias).item()) == 0.0, True)

    def test_initialisation_classifier(self):
        """
        tests whether the weights and bias for non-linear classifier are initialised correctly
        parameters should be different from saved model apart from output bias
        at the same time also tests whether the right parameters are trainable, because otherwise they wouldn't
        be selected in the static method
        remark: shapes of course differ from saved model to saved model but we can expect a saved model with higher
        nr of dimensions than testing model
        """
        pretrained_model_hidden_weight = self._pretrained_model["_hidden.weight"]
        pretrained_model_hidden_bias = self._pretrained_model["_hidden.bias"]
        pretrained_model_output_weight = self._pretrained_model["_output.weight"]
        pretrained_model_output_bias = self._pretrained_model["_output.bias"]

        np.testing.assert_equal(
            pretrained_model_hidden_weight.shape == self.access_named_parameter(self.model, "_hidden.weight").shape,
            False)
        np.testing.assert_equal(
            pretrained_model_hidden_bias.shape == self.access_named_parameter(self.model, "_hidden.bias").shape,
            False)
        np.testing.assert_equal(
            pretrained_model_output_weight.shape == self.access_named_parameter(self.model, "_output.weight").shape,
            False)

        np.testing.assert_equal(
            pretrained_model_output_bias.shape == self.access_named_parameter(self.model, "_output.bias").shape,
            True)

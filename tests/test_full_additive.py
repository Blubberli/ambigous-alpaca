import unittest
import math
import pathlib
import numpy as np
import torch
from torch import optim
from utils import loss_functions
import torch.nn.functional as F
from ranking_models import FullAdditive
from utils.data_loader import PretrainCompmodelDataset
from utils.composition_functions import full_additive


class FullAdditiveTest(unittest.TestCase):
    """
    this class tests the joint matrix model
    This test suite can be ran with:
        python -m unittest -q tests.FullAdditiveTest
    """

    def setUp(self):
        self._data_path_1 = pathlib.Path(__file__).parent.absolute().joinpath("data_pretraining/train.txt")
        self._data_path_2 = pathlib.Path(__file__).parent.absolute().joinpath("data_pretraining/test.txt")
        self._embedding_path = str(pathlib.Path(__file__).parent.absolute().joinpath(
            "embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu"))
        self._dataset_1 = PretrainCompmodelDataset(self._data_path_1, self._embedding_path, head="head",
                                                   mod="modifier", phrase="phrase", separator=" ")
        self._dataset_2 = PretrainCompmodelDataset(self._data_path_2, self._embedding_path, head="head",
                                                   mod="modifier", phrase="phrase", separator=" ")
        modifier_embeddings = F.normalize(torch.rand((50, 100)), dim=1)
        head_embeddings = F.normalize(torch.rand((50, 100)), dim=1)
        gold_composed = F.normalize(torch.rand((50, 100)), dim=1)
        device = torch.device("cpu")
        self.input = {"w1": modifier_embeddings, "w2": head_embeddings, "l": gold_composed, "device": device}
        self.model = FullAdditive(input_dim=100, dropout_rate=0.0, normalize_embeddings=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

    @staticmethod
    def access_named_parameter(model, parameter_name):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == parameter_name:
                    return param.clone()

    def test_model_loss(self):
        self.optimizer.zero_grad()
        composed = self.model(self.input)
        loss = loss_functions.get_loss_cosine_distance(original_phrase=self.input["l"], composed_phrase=composed,
                                                       dim=1, normalize=False).item()
        np.testing.assert_equal(math.isnan(loss), False)

        np.testing.assert_equal(loss >= 0, True)

    def test_parameter_updated_with_training(self):
        noun_matrix_before_training = self.access_named_parameter(self.model, "_adj_matrix")
        adj_matrix_before_training = self.access_named_parameter(self.model, "_noun_matrix")
        for epoch in range(0, 5):
            self.optimizer.zero_grad()
            composed = self.model(self.input)
            loss = loss_functions.get_loss_cosine_distance(original_phrase=self.input["l"], composed_phrase=composed,
                                                           dim=1, normalize=False)
            loss.backward()
            self.optimizer.step()
            noun_matrix_after_training = self.access_named_parameter(self.model, "_adj_matrix")
            adj_matrix_after_training = self.access_named_parameter(self.model, "_noun_matrix")
        difference_noun_matrix = torch.sum(
            noun_matrix_before_training - noun_matrix_after_training).item()
        difference_adj_matrix = torch.sum(
            adj_matrix_before_training - adj_matrix_after_training).item()

        np.testing.assert_equal(difference_noun_matrix != 0.0, True)
        np.testing.assert_equal(difference_adj_matrix != 0.0, True)

    def test_output_shape(self):
        expected_shape = np.array([50, 100])
        composed_phrase = self.model(self.input)
        np.testing.assert_almost_equal(composed_phrase.shape, expected_shape)

    def test_embedding_normalization(self):
        """Test whether the composed phrase has been normalized to unit norm"""
        composed_phrase = self.model(self.input)
        np.testing.assert_almost_equal(np.linalg.norm(composed_phrase[0].data), 1.0)

    def test_composition_function(self):
        matrix_1 = torch.from_numpy(np.full(shape=(2, 2), fill_value=2.0))
        mini_batch_mod = torch.from_numpy(np.array([[1.0, 1.0], [2.0, 2.0]]))

        matrix_2 = torch.from_numpy(np.full(shape=(2, 2), fill_value=1.0))
        mini_batch_head = torch.from_numpy(np.array([[1.0, 1.0], [1.0, 2.0]]))

        comp = full_additive(matrix_1, mini_batch_mod, matrix_2, mini_batch_head)
        result = np.array([[6, 6], [11, 11]])
        np.testing.assert_equal(comp, result)

import unittest
import math
import pathlib
import numpy as np
import torch
from torch import optim
from utils import loss_functions
import torch.nn.functional as F
from ranking_models import TransweighPretrain
from utils.data_loader import StaticRankingDataset, ContextualizedRankingDataset
from torch.utils.data import DataLoader


class PretrainTransweighTest(unittest.TestCase):
    """
    this class tests the training utils script
    This test suite can be ran with:
        python -m unittest -q tests.PretrainTransweighTest
    """

    def setUp(self):
        self._data_path = pathlib.Path(__file__).parent.absolute().joinpath("data_ranking/train.txt")
        self._embedding_path = str(pathlib.Path(__file__).parent.absolute().joinpath(
            "embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu"))
        self._pretrain_dataset = StaticRankingDataset(self._data_path, self._embedding_path, head="head",
                                                      mod="modifier", phrase="phrase", separator=" ")
        self._contextualized_dataset = ContextualizedRankingDataset(data_path="data_ranking/attributes.txt", mod="modifier",
                                                                    head="head", label="label",
                                                                    bert_model='bert-base-german-cased', batch_size=2,
                                                                    lower_case=False, max_len=10, separator=" ",
                                                                    label_definition_path="data_ranking/attribute_definitions")

        modifier_embeddings = F.normalize(torch.rand((50, 100)), dim=1)
        head_embeddings = F.normalize(torch.rand((50, 100)), dim=1)
        gold_composed = F.normalize(torch.rand((50, 100)), dim=1)
        self.input = {"w1": modifier_embeddings, "w2": head_embeddings, "l": gold_composed}

        self.model = TransweighPretrain(input_dim=100, dropout_rate=0.0, normalize_embeddings=True, transformations=10)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

    def test_cosine_distance(self):
        """Test whether the cosine distance is 0 for two equal batches of embeddings"""
        embedding_1 = torch.from_numpy(np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))
        embedding_2 = torch.from_numpy(np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))
        distance = loss_functions.get_loss_cosine_distance(original_phrase=embedding_1, composed_phrase=embedding_2,
                                                           dim=1,
                                                           normalize=True)
        np.testing.assert_equal(distance.item(), 0.0)

    @staticmethod
    def acess_named_parameter(model, parameter_name):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == parameter_name:
                    return param.clone()

    def test_model_loss(self):
        """
        Test whether the composition model can be ran and whether the loss can be computed. The loss should be
        a number larger than zero and not NaN
        """
        self.optimizer.zero_grad()
        composed = self.model(self.input)
        loss = loss_functions.get_loss_cosine_distance(original_phrase=self.input["l"], composed_phrase=composed,
                                                       dim=1, normalize=False).item()
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    def test_parameter_get_updated(self):
        """Test whether initial weight matrices are being updated during training. These parameters should be different
        after training vs before training."""
        tw_tensor_before_training = self.acess_named_parameter(self.model, "_transformation_tensor")
        combining_tensor_before_training = self.acess_named_parameter(self.model, "_combining_tensor")

        for epoch in range(0, 10):
            self.optimizer.zero_grad()
            composed = self.model(self.input)
            loss = loss_functions.get_loss_cosine_distance(original_phrase=self.input["l"], composed_phrase=composed,
                                                           dim=1, normalize=False)
            loss.backward()
            self.optimizer.step()
            tw_tensor_after_training = self.acess_named_parameter(self.model, "_transformation_tensor")
            combining_tensor_after_training = self.acess_named_parameter(self.model, "_combining_tensor")
        difference_combining_tensor = torch.sum(
            combining_tensor_before_training - combining_tensor_after_training).item()
        difference_tw_tensor = torch.sum(tw_tensor_before_training - tw_tensor_after_training).item()
        np.testing.assert_equal(difference_combining_tensor != 0.0, True)
        np.testing.assert_equal(difference_tw_tensor != 0.0, True)

    def test_output_shape(self):
        """Test whether the output shape of the composed representation is as expected"""
        expected_shape = np.array([50, 100])
        composed_phrase = self.model(self.input)
        np.testing.assert_almost_equal(composed_phrase.shape, expected_shape)

    def test_embedding_normalization(self):
        """Test whether the composed phrase has been normalized to unit norm"""
        composed_phrase = self.model(self.input)
        np.testing.assert_almost_equal(np.linalg.norm(composed_phrase[0].data), 1.0)

    def test_dataloader(self):
        """Test whether the pretrained dataset holds a vector for each instance in batch that has the correct
        dimension"""
        dataloader = DataLoader(self._pretrain_dataset, batch_size=3, shuffle=True, num_workers=2)

        data = next(iter(dataloader))
        np.testing.assert_equal(data["w1"].shape, [3, 300])
        np.testing.assert_equal(data["w2"].shape, [3, 300])
        np.testing.assert_equal(data["l"].shape, [3, 300])

    def test_contextualized(self):
        dataloader = DataLoader(self._contextualized_dataset, batch_size=3, shuffle=True, num_workers=2)
        data = next(iter(dataloader))
        np.testing.assert_equal(data["w1"].shape, [3, 768])
        np.testing.assert_equal(data["w2"].shape, [3, 768])
        np.testing.assert_equal(data["l"].shape, [3, 768])
        np.testing.assert_equal(len(data["label"]), 3)

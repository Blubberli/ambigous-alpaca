import unittest
import math
import pathlib
import numpy as np
import json
import torch
from torch import optim
from utils import StaticRankingDataset
from torch.utils.data import DataLoader
from classification_models import MatrixTwoWordClassifier, MatrixTransferClassifier
from ranking_models import MatrixTransferRanker, MatrixPretrain
from utils import training_utils
from utils.loss_functions import multi_class_cross_entropy, get_loss_cosine_distance


class MatrixCompositionModelTest(unittest.TestCase):
    """
    this class tests the matrix composition models
    This test suite can be ran with:
        python -m unittest -q tests.MatrixCompositionModelTest
    """

    def setUp(self):
        config_static = str(pathlib.Path(__file__).parent.absolute().joinpath("test_configs/simple_phrase_config.json"))
        data_pretrain = str(pathlib.Path(__file__).parent.absolute().joinpath("data_pretraining/train.txt"))
        embeddings = str(pathlib.Path(__file__).parent.absolute().joinpath(
            "embeddings/german-structgram-mincount-30-ctx-10-dims-300.fifu"))
        with open(config_static, 'r') as f:
            self.config_static = json.load(f)
        _, _, self.simple_phrase_test = training_utils.get_datasets(self.config_static)
        self.data_loader = DataLoader(dataset=self.simple_phrase_test, batch_size=4)
        self.pretrain_dataset = StaticRankingDataset(data_path=data_pretrain, embedding_path=embeddings,
                                                     separator=" ", head="head", mod="modifier", phrase="phrase")
        self.pretrain_loader = DataLoader(dataset=self.pretrain_dataset, batch_size=4)

        self.model_multiclass = MatrixTwoWordClassifier(input_dim=300, hidden_dim=100, label_nr=3,
                                                        dropout_rate=0.1,
                                                        normalize_embeddings=True)
        self.model_pretrain = MatrixPretrain(input_dim=300, dropout_rate=0.1, normalize_embeddings=True)
        self.train_matrix_classifier()
        self.model_transfer = MatrixTransferClassifier(input_dim=300, hidden_dim=100, label_nr=3, dropout_rate=0.1,
                                                       normalize_embeddings=True,
                                                       pretrained_model="models/matrix_classifier")
        self.train_matrix_pretrain()
        self.model_transfer_rank = MatrixTransferRanker(dropout_rate=0.1, normalize_embeddings=True,
                                                        pretrained_model="models/matrix_pretrain")

    @staticmethod
    def acess_named_parameter(model, parameter_name):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == parameter_name:
                    return param.clone()

    def train_matrix_classifier(self):
        """Auxiliary method to train and save a normal classification matrix model"""
        optimizer = optim.Adam(self.model_multiclass.parameters())
        for batch in self.data_loader:
            optimizer.zero_grad()
            batch["device"] = "cpu"
            output = self.model_multiclass(batch)
            loss = multi_class_cross_entropy(output, batch["l"])
            loss.backward()
            optimizer.step()
        torch.save(self.model_multiclass.state_dict(), "models/matrix_classifier")

    def train_matrix_pretrain(self):
        optimizer = optim.Adam(self.model_pretrain.parameters())
        for batch in self.pretrain_loader:
            batch["device"] = "cpu"
            out = self.model_pretrain(batch).squeeze().to("cpu")
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
            loss.backward()
            optimizer.step()
        torch.save(self.model_pretrain.state_dict(), "models/matrix_pretrain")

    def test_matrix_classifier(self):
        """Test whether matrix classification model can be used to compute the loss"""
        batch = next(iter(self.data_loader))
        batch["device"] = "cpu"
        output = self.model_multiclass(batch)
        loss = multi_class_cross_entropy(output, batch["l"]).item()
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    def test_matrix_pretrain(self):
        """Test whether matrix pretraining model can be used to compute the loss and whether the composed
        representation has the correct shape"""
        batch = next(iter(self.pretrain_loader))
        batch["device"] = "cpu"
        out = self.model_pretrain(batch).squeeze().to("cpu")
        loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"]).item()
        np.testing.assert_equal(out.shape, [4, 300])
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    def test_matrix_transfer(self):
        """Test whether the transfer matrix classification model can be called and whether the loss can be computed"""
        batch = next(iter(self.data_loader))
        batch["device"] = "cpu"
        output = self.model_transfer(batch)
        loss = multi_class_cross_entropy(output, batch["l"]).item()
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    def test_matrix_transfer_ranking(self):
        """Test whether the transfer matrix ranking model can be called and whether the loss can be computed"""
        batch = next(iter(self.pretrain_loader))
        batch["device"] = "cpu"
        out = self.model_transfer_rank(batch).squeeze().to("cpu")
        loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"]).item()
        np.testing.assert_equal(out.shape, [4, 300])
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

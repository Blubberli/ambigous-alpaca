import unittest
import json
from ranking_models import TransweighPretrain, TransweighTransferRanker
from classification_models import TransweighTwoWordClassifier, TransferCompClassifier
from utils import training_utils
import math
import torch
from torch import optim
import pathlib
import numpy as np
from utils.loss_functions import multi_class_cross_entropy, get_loss_cosine_distance
from utils.data_loader import StaticRankingDataset
from torch.utils.data import DataLoader


class TransferCompClassifierTest(unittest.TestCase):
    """
    this class tests the TransferCompClassifier
    This test suite can be ran with:
        python -m unittest -q tests.TransferCompClassifierTest
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

        self.tw_classifier = TransweighTwoWordClassifier(input_dim=300, hidden_dim=50, label_nr=3, dropout_rate=0.1,
                                                         normalize_embeddings=True, transformations=10)
        self.tw_ranker = TransweighPretrain(dropout_rate=0.1,
                                            normalize_embeddings=True, transformations=10, input_dim=300)
        self.train_transweigh_classifier()
        self.train_transweigh_pretrain()
        self.pretrained_model = torch.load("models/tw_classifier")
        self.tw_transfer_classifier = TransferCompClassifier(input_dim=300, hidden_dim=100, label_nr=3,
                                                             dropout_rate=0.1,
                                                             normalize_embeddings=True,
                                                             pretrained_model="models/tw_classifier")
        self.tw_transfer_ranker = TransweighTransferRanker(dropout_rate=0.1,
                                                           normalize_embeddings=True,
                                                           pretrained_model="models/tw_pretrain")
        self.variables = ["_transformation_tensor", "_transformation_bias", "_combining_tensor", "_combining_bias",
                          "_hidden.weight", "_hidden.bias", "_output.weight", "_output.bias"]

    @staticmethod
    def access_named_parameter(model, parameter_name):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == parameter_name:
                    return param.clone()

    def train_transweigh_classifier(self):
        """Auxiliary method to train and save a normal classification matrix model"""
        optimizer = optim.Adam(self.tw_classifier.parameters())
        for batch in self.data_loader:
            optimizer.zero_grad()
            batch["device"] = "cpu"
            output = self.tw_classifier(batch)
            loss = multi_class_cross_entropy(output, batch["l"])
            loss.backward()
            optimizer.step()
        torch.save(self.tw_classifier.state_dict(), "models/tw_classifier")

    def train_transweigh_pretrain(self):
        optimizer = optim.Adam(self.tw_ranker.parameters())
        for batch in self.pretrain_loader:
            batch["device"] = "cpu"
            out = self.tw_ranker(batch).squeeze().to("cpu")
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
            loss.backward()
            optimizer.step()
        torch.save(self.tw_ranker.state_dict(), "models/tw_pretrain")

    def test_tw_transfer_classifier(self):
        """Test whether the transfer matrix classification model can be called and whether the loss can be computed"""
        batch = next(iter(self.data_loader))
        batch["device"] = "cpu"
        output = self.tw_transfer_classifier(batch)
        loss = multi_class_cross_entropy(output, batch["l"]).item()
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    def test_tw_transfer_ranking(self):
        """Test whether the transfer matrix ranking model can be called and whether the loss can be computed"""
        batch = next(iter(self.pretrain_loader))
        batch["device"] = "cpu"
        out = self.tw_transfer_ranker(batch).squeeze().to("cpu")
        loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"]).item()
        np.testing.assert_equal(out.shape, [4, 300])
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

    def test_initialisation_composition(self):
        """
        tests whether the weights and bias for the composition models are initialised correctly
        comp parameters should be the same as in saved model
        at the same time also tests whether the right parameters are trainable, because otherwise they wouldn't
        be selected in the static method
        """

        model_transformation_tensor = self.pretrained_model["_transformation_tensor"]
        model_transformation_bias = self.pretrained_model["_transformation_bias"]
        model_combination_tensor = self.pretrained_model["_combining_tensor"]
        model_combination_bias = self.pretrained_model["_combining_bias"]
        np.testing.assert_equal((torch.sum(self.access_named_parameter(self.tw_transfer_classifier,
                                                                       "_transformation_tensor") -
                                           model_transformation_tensor).item()) == 0.0,
                                True)
        np.testing.assert_equal((torch.sum(
            self.access_named_parameter(self.tw_transfer_classifier, "_transformation_bias") - model_transformation_bias).item()) == 0.0,
                                True)
        np.testing.assert_equal((torch.sum(
            self.access_named_parameter(self.tw_transfer_classifier, "_combining_tensor") - model_combination_tensor).item()) == 0.0,
                                True)
        np.testing.assert_equal((torch.sum(
            self.access_named_parameter(self.tw_transfer_classifier, "_combining_bias") - model_combination_bias).item()) == 0.0, True)

    def test_initialisation_classifier(self):
        """
        tests whether the weights and bias for non-linear classifier are initialised correctly
        parameters should be different from saved model apart from output bias
        at the same time also tests whether the right parameters are trainable, because otherwise they wouldn't
        be selected in the static method
        remark: shapes of course differ from saved model to saved model but we can expect a saved model with higher
        nr of dimensions than testing model
        """
        pretrained_model_hidden_weight = self.pretrained_model["_hidden.weight"]
        pretrained_model_hidden_bias = self.pretrained_model["_hidden.bias"]
        pretrained_model_output_weight = self.pretrained_model["_output.weight"]
        pretrained_model_output_bias = self.pretrained_model["_output.bias"]

        np.testing.assert_equal(
            pretrained_model_hidden_weight.shape == self.access_named_parameter(self.tw_transfer_classifier, "_hidden.weight").shape,
            False)
        np.testing.assert_equal(
            pretrained_model_hidden_bias.shape == self.access_named_parameter(self.tw_transfer_classifier, "_hidden.bias").shape,
            False)
        np.testing.assert_equal(
            pretrained_model_output_weight.shape == self.access_named_parameter(self.tw_transfer_classifier, "_output.weight").shape,
            False)

        np.testing.assert_equal(
            pretrained_model_output_bias.shape == self.access_named_parameter(self.tw_transfer_classifier, "_output.bias").shape,
            True)

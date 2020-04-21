import numpy as np
import torch
import json
from classification_models import BasicTwoWordClassifier
import unittest
import pathlib
from torch.utils.data import DataLoader
from utils import training_utils


class BasicTwoWordClassifierTest(unittest.TestCase):
    """
    this class tests the BasicTwoWordClassifier
    This test suite can be ran with:
        python -m unittest -q tests.BasicTwoWordClassifierTest
    """

    def setUp(self):
        config_static = str(pathlib.Path(__file__).parent.absolute().joinpath("test_configs/simple_phrase_config.json"))
        with open(config_static, 'r') as f:
            self.config_static = json.load(f)
        _, _, self._static_dataset = training_utils.get_datasets(self.config_static)
        self._data = DataLoader(self._static_dataset, batch_size=2)
        self._batch = next(iter(self._data))
        self._batch["device"] = "cpu"

        self.input_dim = 600
        self.hidden_dim = 6
        self.labels = 6

    def test_forward(self):
        """
        tests the classifier implemented in BasicTwoWordClassifier and the overridden method "forward"
        checks whether the output layer is of the right size
        """
        expected_size = torch.tensor(np.zeros((2, 6))).shape
        classifier = BasicTwoWordClassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, label_nr=self.labels,
                                            dropout_rate=0.0)
        res = classifier.forward(self._batch)
        np.testing.assert_allclose(res.shape, expected_size)

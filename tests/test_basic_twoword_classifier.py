import numpy as np
import torch
from scripts import BasicTwoWordClassifier
import unittest
from scripts import SimplePhraseStaticDataset
import pathlib
from torch.utils.data import DataLoader


class BasicTwoWordClassifierTest(unittest.TestCase):
    """
    this class tests the BasicTwoWordClassifier
    This test suite can be ran with:
        python -m unittest -q tests.BasicTwoWordClassifierTest
    """

    def setUp(self):
        self._data_path = pathlib.Path(__file__).parent.absolute().joinpath("data_multiclassification/test.txt")
        self._embedding_path = str(pathlib.Path(__file__).parent.absolute().joinpath(
            "embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu"))
        self._static_dataset = SimplePhraseStaticDataset(self._data_path, self._embedding_path)
        self._data = DataLoader(self._static_dataset, batch_size=2)
        self._batch = next(iter(self._data))

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

import numpy as np
import math
import pathlib
from torch import optim
import unittest
from torch.utils.data import DataLoader
from scripts import PhraseContextClassifier, PhraseAndContextDatasetStatic
from scripts import multi_class_cross_entropy


class PhraseContextClassifierTest(unittest.TestCase):
    """
    this class tests the PhraseContextClassifier
    This test suite can be ran with:
        python -m unittest -q tests.PhraseContextClassifierTest
    """

    def setUp(self):
        self._data_path = pathlib.Path(__file__).parent.absolute().joinpath("data_multiclassification/test.txt")
        self._embedding_path = str(pathlib.Path(__file__).parent.absolute().joinpath(
            "embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu"))

        self._static_dataset = PhraseAndContextDatasetStatic(self._data_path, self._embedding_path,
                                                             tokenizer_model="de_CMC")
        self.model = PhraseContextClassifier(embedding_dim=300, hidden_size=200, dropout_rate=0.0,
                                             forward_hidden_dim=100, label_nr=3, num_layers=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

    def test_model_multiclass(self):
        """
        Test whether the multiclass classifier can be ran and whether the loss can be computed. The loss should be
        a number larger than zero and not NaN
        """
        data_loader = DataLoader(self._static_dataset,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=0)

        for batch in data_loader:
            # context is a list of list of word embeddings
            batch["device"] = "cpu"
            out = self.model(batch, True).squeeze()
            loss = multi_class_cross_entropy(out, batch["l"])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            break
        loss = loss.data.numpy()
        np.testing.assert_equal(math.isnan(loss), False)
        np.testing.assert_equal(loss >= 0, True)

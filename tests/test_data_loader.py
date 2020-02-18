import pathlib
import unittest
import numpy as np
from torch.utils.data import DataLoader
from scripts.data_loader import SimplePhraseContextualizedDataset, SimplePhraseStaticDataset, SimplePhraseDataset


class DataLoaderTest(unittest.TestCase):
    """
    This class tests the Datasets that can be used with the torch DataLoader.
    This test suite can be ran with:
        python -m unittest -q tests.DataLoaderTest
    """

    def setUp(self):
        self._data_path = pathlib.Path(__file__).parent.absolute().joinpath("data_multiclassification/test.txt")
        self._embedding_path = str(pathlib.Path(__file__).parent.absolute().joinpath(
            "embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu"))

        self._contextualized_dataset = SimplePhraseContextualizedDataset(self._data_path, 'bert-base-german-cased', 20,
                                                                         False, 20)
        self._static_dataset = SimplePhraseStaticDataset(self._data_path, self._embedding_path)

    def test_exception(self):
        """ Trying to initialize the abstract Dataset should throw a Type Exception """
        self.assertRaises(TypeError, lambda: SimplePhraseDataset(self._data_path))

    def test_shape(self):
        """Test whether the word1 and word2 embeddings have the shape (batchsize, embedding dim) and  the
        labels have the shape (batchsize)
        """
        dataloader = DataLoader(self._contextualized_dataset, batch_size=5, shuffle=True, num_workers=2)
        w1, w2, l = next(iter(dataloader))
        np.testing.assert_equal(np.array(w1.shape), [5, 768])
        np.testing.assert_equal(np.array(w2.shape), [5, 768])
        np.testing.assert_equal(np.array(l.shape), [5])
        dataloader = DataLoader(self._static_dataset, batch_size=5, shuffle=True, num_workers=2)
        w1, w2, l = next(iter(dataloader))
        np.testing.assert_equal(np.array(w1.shape), [5, 300])
        np.testing.assert_equal(np.array(w2.shape), [5, 300])
        np.testing.assert_equal(np.array(l.shape), [5])

    def test_labels(self):
        """Test whether the label dictionary contains 3 different labels for the multiclass classification dataset"""
        np.testing.assert_equal(len(self._contextualized_dataset.label2index), 3)

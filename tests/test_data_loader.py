import pathlib
import unittest
import json
import numpy as np
from torch.utils.data import DataLoader
from utils import training_utils
from utils import SimplePhraseDataset


class DataLoaderTest(unittest.TestCase):
    """
    This class tests the Datasets that can be used with the torch DataLoader.
    This test suite can be ran with:
        python -m unittest -q tests.DataLoaderTest
    """

    def setUp(self):
        config_static = str(pathlib.Path(__file__).parent.absolute().joinpath("test_configs/simple_phrase_config.json"))
        config_contextualized = str(
            pathlib.Path(__file__).parent.absolute().joinpath("test_configs/simple_phrase_contextualized_config.json"))
        with open(config_static, 'r') as f:
            self.config_static = json.load(f)
        with open(config_contextualized, 'r') as f:
            self.config_contextualized = json.load(f)
        _, _, self.simple_phrase_test = training_utils.get_datasets(self.config_static)
        _, _, self.contextualized_test = training_utils.get_datasets(self.config_contextualized)

    def test_exception(self):
        """ Trying to initialize the abstract Dataset should throw a Type Exception """
        self.assertRaises(TypeError, lambda: SimplePhraseDataset(self.config_static["test_data_path"], label="label",
                                                                 label_encoder=None, phrase="phrase", separator=" "))

    def test_shape(self):
        """Test whether the word1 and word2 embeddings have the shape (batchsize, embedding dim) and  the
        labels have the shape (batchsize)
        """
        dataloader = DataLoader(self.contextualized_test, batch_size=5, shuffle=True, num_workers=2)

        data = next(iter(dataloader))

        np.testing.assert_equal(np.array(data["w1"].shape), [5, 768])
        np.testing.assert_equal(np.array(data["w2"].shape), [5, 768])
        np.testing.assert_equal(np.array(data["l"].shape), [5])
        dataloader = DataLoader(self.simple_phrase_test, batch_size=5, shuffle=True, num_workers=2)
        data = next(iter(dataloader))
        np.testing.assert_equal(np.array(data["w1"].shape), [5, 300])
        np.testing.assert_equal(np.array(data["w2"].shape), [5, 300])
        np.testing.assert_equal(np.array(data["l"].shape), [5])

    def test_labels(self):
        """Test whether the label dictionary contains 3 different labels for the multiclass classification dataset"""
        np.testing.assert_equal(len(self.contextualized_test.label2index), 3)

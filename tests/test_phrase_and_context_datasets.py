import unittest
import pathlib
import json
import numpy as np
from utils import training_utils
from torch.utils.data import DataLoader


class PhraseAndContextDatasetsTest(unittest.TestCase):
    def setUp(self):
        config_static = str(
            pathlib.Path(__file__).parent.absolute().joinpath("test_configs/phrase_context_static_config.json"))
        config_contextualized = str(
            pathlib.Path(__file__).parent.absolute().joinpath("test_configs/phrase_context_contextualized_config.json"))
        with open(config_static, 'r') as f:
            self.config_static = json.load(f)
        with open(config_contextualized, 'r') as f:
            self.config_contextualized = json.load(f)
        _, _, self.static_set = training_utils.get_datasets(self.config_static)
        _, _, self.bert_set = training_utils.get_datasets(self.config_contextualized)

    def test_dataloader(self):
        train_loader = DataLoader(self.static_set,
                                  batch_size=10,
                                  shuffle=True,
                                  num_workers=0)
        for batch in train_loader:
            batch["device"] = "cpu"
            max_len = batch["seq"].shape[1]
            # shape of sequence has three dimensions: batchsize, max_seq_len, embedding_dim
            np.testing.assert_equal(len(batch["seq"].shape), 3)
            np.testing.assert_equal(batch["w1"].shape[0], batch["w2"].shape[0])
            np.testing.assert_equal(batch["w1"].shape[0], batch["seq"].shape[0])
            np.testing.assert_equal(batch["seq_lengths"].shape[0], batch["l"].shape[0])
            max_val = max(batch["seq_lengths"]).data.numpy()
            np.testing.assert_equal(max_len >= max_val, True)

        train_loader = DataLoader(self.bert_set,
                                  batch_size=10,
                                  shuffle=True,
                                  num_workers=0)
        for batch in train_loader:
            batch["device"] = "cpu"
            max_len = batch["seq"].shape[1]
            # shape of sequence has three dimensions: batchsize, max_seq_len, embedding_dim
            np.testing.assert_equal(len(batch["seq"].shape), 3)
            np.testing.assert_equal(batch["w1"].shape[0], batch["w2"].shape[0])
            np.testing.assert_equal(batch["w1"].shape[0], batch["seq"].shape[0])
            np.testing.assert_equal(batch["seq_lengths"].shape[0], batch["l"].shape[0])
            max_val = max(batch["seq_lengths"]).data.numpy()
            np.testing.assert_equal(max_len >= max_val, True)

    def test_padding(self):
        train_loader = DataLoader(self.static_set,
                                  batch_size=10,
                                  shuffle=True,
                                  num_workers=0)
        for batch in train_loader:
            batch["device"] = "cpu"
            max_len = batch["seq"].shape[1]
            max_val = max(batch["seq_lengths"]).data.numpy()
            if max_len != max_val:
                padded_vec = batch["seq"][0]
                real_len = batch["seq_lengths"][0].data.numpy()
                non_padded = np.count_nonzero(padded_vec, axis=0)[0]
                np.testing.assert_equal(real_len, non_padded)
                np.testing.assert_array_almost_equal(padded_vec[max_len - 1], np.zeros(shape=[300]))

        train_loader = DataLoader(self.bert_set,
                                  batch_size=10,
                                  shuffle=True,
                                  num_workers=0)
        for batch in train_loader:
            batch["device"] = "cpu"
            max_len = batch["seq"].shape[1]
            max_val = max(batch["seq_lengths"]).data.numpy()
            if max_len != max_val:
                padded_vec = batch["seq"][0]
                real_len = batch["seq_lengths"][0].data.numpy()
                non_padded = np.count_nonzero(padded_vec, axis=0)[0]
                np.testing.assert_equal(real_len, non_padded)
                np.testing.assert_array_almost_equal(padded_vec[max_len - 1], np.zeros(shape=[768]))

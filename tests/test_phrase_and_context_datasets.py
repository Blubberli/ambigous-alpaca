import unittest
import pathlib
from scripts import PhraseAndContextDatasetStatic, PhraseAndContextDatasetBert
import numpy as np
from torch.utils.data import DataLoader



class PhraseAndContextDatasetsTest(unittest.TestCase):
    def setUp(self):
        data_path = pathlib.Path(__file__).parent.absolute().joinpath("data_multiclassification/train.txt")
        embedding_path = str(pathlib.Path(__file__).parent.absolute().joinpath(
            "embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu"))
        self.static_set = PhraseAndContextDatasetStatic(data_path=data_path, embedding_path=embedding_path,
                                                   tokenizer_model="de_CMC")
        self.bert_set = PhraseAndContextDatasetBert(data_path=data_path, bert_model='bert-base-german-cased', max_len=20,
                                              lower_case=False, batch_size=20, tokenizer_model="de_CMC")


    def test_dataloader(self):
        train_loader = DataLoader(self.static_set,
                                  batch_size=10,
                                  shuffle=True,
                                  num_workers=0)
        for batch in train_loader:
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
            max_len = batch["seq"].shape[1]
            max_val = max(batch["seq_lengths"]).data.numpy()
            if max_len != max_val:
                padded_vec = batch["seq"][0]
                real_len = batch["seq_lengths"][0].data.numpy()
                non_padded = np.count_nonzero(padded_vec, axis=0)[0]
                np.testing.assert_equal(real_len, non_padded)
                np.testing.assert_array_almost_equal(padded_vec[max_len-1], np.zeros(shape=[300]))
        train_loader = DataLoader(self.bert_set,
                                  batch_size=10,
                                  shuffle=True,
                                  num_workers=0)
        for batch in train_loader:
            max_len = batch["seq"].shape[1]
            max_val = max(batch["seq_lengths"]).data.numpy()
            if max_len != max_val:
                padded_vec = batch["seq"][0]
                real_len = batch["seq_lengths"][0].data.numpy()
                non_padded = np.count_nonzero(padded_vec, axis=0)[0]
                np.testing.assert_equal(real_len, non_padded)
                np.testing.assert_array_almost_equal(padded_vec[max_len - 1], np.zeros(shape=[768]))


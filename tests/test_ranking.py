import unittest
import torch
import numpy as np
from scripts.data_loader import extract_all_words, PretrainCompmodelDataset
from torch.utils.data import DataLoader
from scripts.ranking import Ranker


class RankingTest(unittest.TestCase):
    """
    This class tests the functionality of the Ranker.
    This test suite can be ran with:
        python -m unittest -q tests.RankingTest
    """

    def setUp(self):
        # test_data_dir = Path(__file__).resolve().parents[1]
        self.predictions = "embeddings/predictions_rank.npy"
        self.embedding_path = "embeddings/ranking_embeddings.fifu"
        self.ranks_path = "data_pretraining/ranks.txt"
        self.labels = extract_all_words("data_pretraining/test.txt", "data_pretraining/test.txt",
                                        "data_pretraining/test.txt", separator=" ", head="head", modifier="modifier",
                                        phrase="phrase")

        dataset_test = PretrainCompmodelDataset(data_path="data_pretraining/test.txt",
                                                embedding_path=self.embedding_path, separator=" ",
                                                phrase="phrase", mod="modifier", head="head")
        self.data_loader = DataLoader(dataset=dataset_test, shuffle=False, batch_size=len(dataset_test))

    def test_all_ranks(self):
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks="data_pretraining/ranks.txt",
                        embedding_path=self.embedding_path, all_labels=self.labels,
                        data_loader=self.data_loader, max_rank=14, y_label="phrase")
        ranks = ranker._ranks
        np.testing.assert_equal(ranks, np.full([4], 1))

    def test_bad_prediction(self):
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks=self.ranks_path,
                        embedding_path=self.embedding_path, all_labels=self.labels,
                        data_loader=self.data_loader, max_rank=5, y_label="phrase")
        bad_prediction = torch.from_numpy(np.array([0.0, 1.0]))
        ranker._predicted_embeddings[2] = bad_prediction
        ranks = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks[2], 5)

    def test_close_prediction(self):
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks=self.ranks_path,
                        embedding_path=self.embedding_path, all_labels=self.labels,
                        data_loader=self.data_loader, max_rank=5, y_label="phrase")
        close_prediction = torch.from_numpy(np.array([0.7030, 0.668]))
        ranker._predicted_embeddings[2] = close_prediction / np.linalg.norm(close_prediction)
        ranks = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks[2], 2)

    def test_ranks(self):
        predictions = np.array([[0.3426, 0.9395], [0.7465, 0.6654],
                                [0.9788, 0.2049], [0.9788, 0.2049]], dtype=np.float32)

        correct_ranks = [1, 1, 5, 6]
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks=self.ranks_path,
                        embedding_path=self.embedding_path,
                        all_labels=self.labels,
                        data_loader=self.data_loader, max_rank=6, y_label="phrase")
        ranker._predicted_embeddings = predictions

        ranks = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks, correct_ranks)

    def test_quartiles_uneven(self):
        ranks = [6, 7, 15, 36, 39, 40, 41, 42, 43, 47, 49]
        quartiles, percent = Ranker.calculate_quartiles(ranks)
        result = [15, 40, 43]
        np.testing.assert_equal(quartiles, result)

    def test_quartiles_even(self):
        ranks = [7, 15, 36, 39, 40, 41]
        quartiles, percent = Ranker.calculate_quartiles(ranks)
        result = [15, 37.5, 40]
        np.testing.assert_equal(quartiles, result)

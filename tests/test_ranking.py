import unittest
from pathlib import Path
import ffp
import torch
import numpy as np
from scripts.data_loader import extract_all_words, PretrainCompmodelDataset
from torch.utils.data import DataLoader
from scripts.ranking import Ranker
from scripts import StaticEmbeddingExtractor


class EvaluationTest(unittest.TestCase):
    """
    This class tests the functionality of the evaluation.py script.
    This test suite can be ran with:
        python -m unittest -q tests.EvaluationTest
    """

    def setUp(self):
        # test_data_dir = Path(__file__).resolve().parents[1]
        self.predictions = "/Users/nwitte/PycharmProjects/ambigous-alpaca/tests/embeddings/predictions_rank.npy"
        embedding_path = "embeddings/ranking_embeddings.fifu"
        self.labels = extract_all_words("data_pretraining/train.txt", "data_pretraining/test.txt",
                                        "data_pretraining/train.txt", separator=" ", head="head", modifier="modifier",
                                        phrase="phrase")

        dataset_test = PretrainCompmodelDataset(data_path="data_pretraining/test.txt",
                                                embedding_path=embedding_path, separator=" ",
                                                phrase="phrase", mod="modifier", head="head")
        self.data_loader = DataLoader(dataset=dataset_test, shuffle=False, batch_size=len(dataset_test))

    def test_all_ranks(self):
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks="data_pretraining/ranks.txt",
                        embedding_path="embeddings/ranking_embeddings.fifu", all_labels=self.labels,
                        dataloader=self.data_loader, max_rank=14)
        ranks = ranker._ranks
        np.testing.assert_equal(ranks, np.full([8], 1))

    def test_n_nearest(self):
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks="data_pretraining/ranks.txt",
                        embedding_path="embeddings/ranking_embeddings.fifu", all_labels=self.labels,
                        dataloader=self.data_loader, max_rank=5)
        ranks = ranker._ranks
        np.testing.assert_equal(ranks, np.full([8], 1))

    def test_bad_prediction(self):
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks="data_pretraining/ranks.txt",
                        embedding_path="embeddings/ranking_embeddings.fifu", all_labels=self.labels,
                        dataloader=self.data_loader, max_rank=5)
        bad_prediction = torch.from_numpy(np.array([0.0, 1.0]))
        ranker._predicted_embeddings[2] = bad_prediction
        ranks = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks[2], 5)

    def test_close_prediction(self):
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks="data_pretraining/ranks.txt",
                        embedding_path="embeddings/ranking_embeddings.fifu", all_labels=self.labels,
                        dataloader=self.data_loader, max_rank=5)
        close_prediction = torch.from_numpy(np.array([0.7030, 0.668]))
        ranker._predicted_embeddings[2] = close_prediction / np.linalg.norm(close_prediction)
        ranks = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks[2], 2)

    def test_ranks(self):
        predictions = np.array([[0.88143742, 0.47230083], [0.83301994, 0.55324296],
                                [0.99520118, 0.09784996], [0.50229784, 0.86469467]], dtype=np.float32)

        compounds = ["apfelbaum", "quarkstrudel", "kirschbaum", "kirschstrudel"]
        correct_ranks = [5, 4, 14, 14]
        ranker = Ranker(path_to_predictions=self.predictions,
                        path_to_ranks="data_pretraining/ranks.txt",
                        embedding_path="embeddings/ranking_embeddings.fifu", all_labels=self.labels,
                        dataloader=self.data_loader, max_rank=14)
        ranker._predicted_embeddings = predictions
        ranker._true_labels = compounds
        label2index = dict(zip(self.labels, range(len(self.labels))))
        embedder = StaticEmbeddingExtractor("embeddings/ranking_embeddings.fifu")
        embeddings = embedder.get_array_embeddings(self.labels)
        ranks = ranker.get_target_based_rank()
        print("ranks")
        print(ranks)
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


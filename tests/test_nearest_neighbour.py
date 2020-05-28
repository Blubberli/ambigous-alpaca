import unittest
import torch
import numpy as np
from utils.data_loader import extract_all_words, StaticRankingDataset, ContextualizedRankingDataset, \
    extract_all_labels
from torch.utils.data import DataLoader
from training_scripts.nearest_neighbour import NearestNeigbourRanker
from utils import StaticEmbeddingExtractor, BertExtractor


class RankingTest(unittest.TestCase):
    """
    This class tests the functionality of the Ranker.
    This test suite can be ran with:
        python -m unittest -q tests.RankingTest
    """

    def setUp(self):
        self.predictions = "embeddings/predictions_rank.npy"
        self.embedding_path = "embeddings/ranking_embeddings.fifu"
        self.ranks_path = "data_ranking/ranks.txt"
        self.labels = extract_all_words("data_ranking/test.txt", "data_ranking/test.txt",
                                        "data_ranking/test.txt", separator=" ", head="head", modifier="modifier",
                                        phrase="phrase")
        self.static_extractor = StaticEmbeddingExtractor(self.embedding_path)
        self.contextualized_extractor = BertExtractor('bert-base-german-cased', 20, False, 4)

        dataset_test = StaticRankingDataset(data_path="data_ranking/test.txt",
                                            embedding_path=self.embedding_path, separator=" ",
                                            phrase="phrase", mod="modifier", head="head")

        self.attributes = extract_all_labels(test_data="data_ranking/attributes.txt",
                                             training_data="data_ranking/attributes.txt",
                                             validation_data="data_ranking/attributes.txt", separator=" ",
                                             label="label")
        contextualized_dataset = ContextualizedRankingDataset(data_path="data_ranking/attributes.txt",
                                                              mod="modifier",
                                                              head="head", label="label",
                                                              bert_model='bert-base-german-cased', batch_size=2,
                                                              lower_case=False, max_len=10, separator=" ",
                                                              label_definition_path="data_ranking/attribute_definitions")

        self.data_loader = DataLoader(dataset=dataset_test, shuffle=False, batch_size=len(dataset_test))
        self.data_loader_contextualized = DataLoader(dataset=contextualized_dataset, shuffle=False,
                                                     batch_size=len(contextualized_dataset))

    def test_all_ranks(self):
        ranker = NearestNeigbourRanker(path_to_predictions=self.predictions,
                                       embedding_extractor=self.static_extractor, all_labels=self.labels,
                                       data_loader=self.data_loader, max_rank=14, y_label="phrase")
        ranks = ranker._ranks
        np.testing.assert_equal(ranks, np.full([4], 1))

    def test_bad_prediction(self):
        ranker = NearestNeigbourRanker(path_to_predictions=self.predictions,
                                       embedding_extractor=self.static_extractor, all_labels=self.labels,
                                       data_loader=self.data_loader, max_rank=4, y_label="phrase")
        bad_prediction = torch.from_numpy(np.array([0.0, 1.0]))
        ranker._predicted_embeddings[2] = bad_prediction
        ranks, _, _ = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks[2], 4)

    def test_close_prediction(self):
        ranker = NearestNeigbourRanker(path_to_predictions=self.predictions,
                                       embedding_extractor=self.static_extractor, all_labels=self.labels,
                                       data_loader=self.data_loader, max_rank=5, y_label="phrase")
        close_prediction = torch.from_numpy(np.array([0.7030, 0.668]))
        ranker._predicted_embeddings[2] = close_prediction / np.linalg.norm(close_prediction)
        ranks, _, _ = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks[2], 2)

    def test_ranks(self):
        predictions = np.array([[0.3426, 0.9395], [0.7465, 0.6654],
                                [0.9788, 0.2049], [0.9788, 0.2049]], dtype=np.float32)

        correct_ranks = [3, 4, 3, 6]
        ranker = NearestNeigbourRanker(path_to_predictions=self.predictions,
                                       embedding_extractor=self.static_extractor,
                                       all_labels=self.labels,
                                       data_loader=self.data_loader, max_rank=6, y_label="phrase")
        ranker._predicted_embeddings = predictions
        ranks, _, _ = ranker.get_target_based_rank()
        np.testing.assert_equal(ranks, correct_ranks)

    def test_nearestneighbour_contextualized(self):
        predictions = "embeddings/bert_predictions.npy"
        correct_ranks = [1, 2, 2, 2]
        ranker = NearestNeigbourRanker(path_to_predictions=predictions,
                                       embedding_extractor=self.contextualized_extractor,
                                       all_labels=self.attributes,
                                       data_loader=self.data_loader_contextualized, max_rank=6, y_label="label")
        ranks = ranker.ranks
        np.testing.assert_equal(ranks, correct_ranks)

    def test_precicion_at_rank(self):
        ranks = [1, 4, 5, 2, 6, 7, 8, 10, 15, 1]
        p_at_1 = NearestNeigbourRanker.precision_at_rank(1, ranks)
        p_at_5 = NearestNeigbourRanker.precision_at_rank(5, ranks)
        np.testing.assert_equal(p_at_1, 0.2)
        np.testing.assert_equal(p_at_5, 0.5)

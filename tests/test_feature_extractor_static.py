import unittest
from scripts import StaticEmbeddingExtractor
import numpy as np


class StaticEmbeddingExtractorTest(unittest.TestCase):

    def setUp(self):
        path_skipgram = "embeddings/german-skipgram-mincount-30-ctx-10-dims-300.fifu"
        path_struct = "embeddings/german-structgram-mincount-30-ctx-10-dims-300.fifu"
        path_word2vec = "/Users/ehuber/Documents/ambiguous_alpaca/ambigous-alpaca/Data/Embeddings/English/encow-adj-n.fifu"
        self.extractor_skipgram = StaticEmbeddingExtractor(path_skipgram)
        self.extractor_structgram = StaticEmbeddingExtractor(path_struct)
        self.extractor_word2vec = StaticEmbeddingExtractor(path_word2vec)
        self.emb_skip = self.extractor_skipgram.get_embedding("Pferd")
        self.emb_struct = self.extractor_structgram.get_embedding("Pferd")
        array_words = ["Schaaf", "Pferd", "Hund", "Katze"]
        self.array_embeddings = self.extractor_skipgram.get_array_embeddings(array_words)

    def test_dimension(self):
        """
        test dimension of both types of embedding (skipgram and structured skipgrams)
        """
        expected = 300

        np.testing.assert_equal(expected, self.emb_skip.shape[0])
        np.testing.assert_equal(expected, self.emb_struct.shape[0])

    def test_embeddings_different(self):
        """
        tests that two different embeddings (structured vs. normal skipgram) correspond to the the same word
        """
        np.testing.assert_equal(np.array_equal(self.emb_skip, self.emb_struct), False)

    def test_array_embedding(self):
        """
        tests length of array of embeddings and dimension of embedding
        """
        expected = 4
        expected_dimensions = 300
        np.testing.assert_equal(expected, len(self.array_embeddings))
        np.testing.assert_equal(expected_dimensions, self.array_embeddings[0].shape[0])
    def test_additional(self):

        emb = self.extractor_word2vec.get_embedding("hobadibuduasdfasdfadsdsaadsdfsdfs")
        np.testing.assert_equal(emb.shape[0], self.extractor_word2vec.embedding_dim)
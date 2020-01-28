import unittest
from scripts import StaticEmbeddingExtractor


class StaticEmbeddingExtractorTest(unittest.TestCase):

    def setUp(self):
        path_skipgram = "/Users/ehuber/Documents/ambiguous_alpaca/ambigous-alpaca/german-skipgram-mincount-30-ctx-10-dims-300.fifu"
        path_struct = "/Users/ehuber/Documents/ambiguous_alpaca/ambigous-alpaca/german-structgram-mincount-30-ctx-10-dims-300.fifu"
        self.extractor_skipgram = StaticEmbeddingExtractor(path_skipgram)
        self.extractor_structgram = StaticEmbeddingExtractor(path_struct)

    def test_dimension(self):
        expected = 300
        emb = self.extractor_skipgram.get_embedding("Pferd")
        #...
    def test_distance(self):
        pass


    #...
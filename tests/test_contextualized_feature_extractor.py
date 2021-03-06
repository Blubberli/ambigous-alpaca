import pathlib
import numpy as np
import unittest
import pandas as pd
from scipy.spatial.distance import cosine
from utils import BertExtractor


class BertExtractorTest(unittest.TestCase):
    """
    this class tests the BertExtractor
    This test suite can be ran with:
        python -m unittest -q tests.BertExtractorTest
    """

    def setUp(self):
        self.extractor = BertExtractor('bert-base-german-cased', 20, False, 4)
        s1 = "Der Junge sitzt auf der Bank im Park"
        s2 = "Ich gehe zur Bank um Geld abzuheben"
        s3 = "In der Kirche steht eine Bank aus Holz auf der die Leute sitzen können"
        s4 = "Der Vater arbeitet bei der Bank als Finanzchef"
        self.adj_noun_phrases = ["stürmisch Zustimmung", "rot Kleid"]
        self.attributes = ["Emotion", "Farbe"]
        self.definitions = ["Emotion beschreibt die Gemütsbewegung, Gefühl .",
                            "Farbe ist die durch Lichtstrahlen bestimmter Wellenlänge hervorgerufene Erscheinung vor "
                            "dem Auge ."]
        self.modifier = ["stürmisch", "rot"]
        self.sentences = [s1, s2, s3, s4]
        self.target_words = ["Bank", "Bank", "Bank", "Bank"]
        data_path = pathlib.Path(__file__).parent.absolute().joinpath("data_multiclassification/train.txt")
        test_data = pd.read_csv(data_path, sep="\t")
        self.sentence_list = list(test_data["context"])
        self.target_word_list = list(test_data["phrase"])

    def test_subword_indices(self):
        """
        Tests whether the correct index of a word within a tokenized sentence will be returned. Because 'Ich' will not
        be tokenized into word pieces it will be the first word of the sentence and will have an index of 1 (The
        special sentence start token will be index 0)
        """
        indices = self.extractor.word_indices_in_sentence(sentence=self.sentences[1], target_word="Ich")
        np.testing.assert_almost_equal(indices, [1])
        indices = self.extractor.word_indices_in_sentence(sentence=self.sentences[1], target_word="abzuheben")
        np.testing.assert_almost_equal(indices, [7, 8])

    def test_convert_sentence_to_indices(self):
        """
        Tests whether a sentence is correctly converted into its indices arrays. Input_ids will be an array of word
        piece indices padded with zeros until a maximum length of 10. Token_type_ids will be ann arrays of zeros,
        because we assume that we are only dealing with one-sentence classification tasks. Attention_mask will be
        an array of 6 ones, indicating that these words are actual words and 4 zeros, because every sentence is padded
        to a length of 10.
        """
        sentence = "Das ist ein Satz"
        result = self.extractor.convert_sentence_to_indices(sentence)
        input_ids = result["input_ids"]
        token_type_ids = result["token_type_ids"]
        attention_mask = result["attention_mask"]
        np.testing.assert_equal(input_ids[:6] != 0, True)
        np.testing.assert_almost_equal(token_type_ids, np.zeros(shape=[20, ]))
        np.testing.assert_almost_equal(attention_mask[:6], np.ones(shape=[6, ]))
        np.testing.assert_almost_equal(attention_mask[6:], np.zeros(shape=[14, ]))
        np.testing.assert_equal(len(token_type_ids), len(attention_mask))
        np.testing.assert_equal(len(input_ids), len(attention_mask))

    def test_single_word_embedding(self):
        """Test whether a single embedding of a longer word can be extracted from a sentence and has the correct dim."""
        target_word_indices = self.extractor.word_indices_in_sentence(self.sentences[3], "Finanzchef")
        batch_input_ids, batch_token_type_ids, batch_attention_mask = self.extractor.convert_sentence_batch_to_indices(
            [self.sentences[3]])
        last_hidden_states, all_layers = self.extractor.get_bert_vectors(batch_input_ids, batch_token_type_ids,
                                                                         batch_attention_mask)
        single_embedding = self.extractor.get_single_word_embedding(last_hidden_states[0], target_word_indices)
        np.testing.assert_equal(single_embedding.shape, [768])

    def test_batch_sentences(self):
        """
        Check whether contextualized embeddings can be computed for a list of sentences. Check also whether the
        representations are more similar to a word that is clearly of the same sense vs. the case when they clearly
        are not of the same sense.
        """
        result = self.extractor.get_single_word_representations(self.sentences, self.target_words)
        np.testing.assert_equal(result.shape[0], 4)
        np.testing.assert_equal(result.shape[1], 768)
        import torch.nn as nn
        cos = nn.CosineSimilarity(dim=0)
        bank_s1 = result[0]
        bank_s2 = result[1]
        bank_s3 = result[2]
        bank_s4 = result[3]
        np.testing.assert_equal(cos(bank_s1, bank_s3).item() > cos(bank_s1, bank_s4).item(), True)
        np.testing.assert_equal(cos(bank_s2, bank_s4).item() > cos(bank_s2, bank_s3).item(), True)

    def test_mean_pooling(self):
        """Test whether we can extract a mean of all layers, resulting in one layer for each token and a mean for all
            token embeddings to extract a single sentence vector"""
        batch_input_ids, batch_token_type_ids, batch_attention_mask = self.extractor.convert_sentence_batch_to_indices(
            [self.sentences[3]])
        last_layer, hidden_layers = self.extractor.get_bert_vectors(batch_input_ids, batch_attention_mask,
                                                                    batch_token_type_ids)
        layer_mean = self.extractor.get_mean_layer_pooling(hidden_layers, 0, 12)
        sentence_mean = self.extractor.get_mean_sentence_pooling(last_layer)
        np.testing.assert_almost_equal(layer_mean.shape, [1, 20, 768])
        np.testing.assert_almost_equal(sentence_mean.shape, [1, 768])

    def test_feature_extractor_dataset(self):
        """Test whether we can extract word embeddings for a larger list of sentences and words"""
        contextualized_embeddings = self.extractor.get_single_word_representations(self.sentence_list,
                                                                                   self.target_word_list)
        np.testing.assert_equal(contextualized_embeddings.shape[0], 105)
        np.testing.assert_equal(contextualized_embeddings.shape[1], 768)

    def test_simple_phrases(self):
        attribute_embeddings = self.extractor.get_single_word_representations(target_words=self.attributes,
                                                                              sentences=self.definitions)
        contextualized_adjectives = self.extractor.get_single_word_representations(sentences=self.adj_noun_phrases,
                                                                                   target_words=self.modifier)
        dunkel_machenschaft_bewertung = 1 - cosine(attribute_embeddings[0], contextualized_adjectives[0])
        dunkel_machenschaft_farbe = 1 - cosine(attribute_embeddings[1], contextualized_adjectives[0])

        rot_kleid_bewertung = 1 - cosine(attribute_embeddings[0], contextualized_adjectives[1])
        rot_kleid_farbe = 1 - cosine(attribute_embeddings[1], contextualized_adjectives[1])

        np.testing.assert_equal(dunkel_machenschaft_bewertung > dunkel_machenschaft_farbe, True)
        np.testing.assert_equal(rot_kleid_bewertung < rot_kleid_farbe, True)

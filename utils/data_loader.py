from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import pandas
import torch
from torch import nn
import somajo
import numpy as np
from utils import BertExtractor, StaticEmbeddingExtractor


def create_label_encoder(all_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    return label_encoder


def extract_all_labels(training_data, validation_data, test_data, separator, label):
    training_labels = set(pandas.read_csv(training_data, delimiter=separator, index_col=False)[label])
    validation_labels = set(pandas.read_csv(validation_data, delimiter=separator, index_col=False)[label])
    test_labels = set(pandas.read_csv(test_data, delimiter=separator, index_col=False)[label])
    all_labels = list(training_labels.union(validation_labels).union(test_labels))
    return all_labels


def extract_all_words(training_data, validation_data, test_data, separator, modifier, head, phrase):
    training_labels = set(pandas.read_csv(training_data, delimiter=separator, index_col=False)[modifier]).union(
        set(pandas.read_csv(training_data, delimiter=separator, index_col=False)[head])).union(
        set(pandas.read_csv(training_data, delimiter=separator, index_col=False)[phrase]))
    validation_labels = set(pandas.read_csv(validation_data, delimiter=separator, index_col=False)[modifier]).union(
        set(pandas.read_csv(validation_data, delimiter=separator, index_col=False)[head])).union(
        set(pandas.read_csv(validation_data, delimiter=separator, index_col=False)[phrase]))
    test_labels = set(pandas.read_csv(test_data, delimiter=separator, index_col=False)[modifier]).union(
        set(pandas.read_csv(test_data, delimiter=separator, index_col=False)[head])).union(
        set(pandas.read_csv(test_data, delimiter=separator, index_col=False)[phrase]))
    all_labels = list(training_labels.union(validation_labels).union(test_labels))
    return all_labels


class SimplePhraseDataset(ABC, Dataset):
    def __init__(self, data_path, label_encoder, separator, phrase, label):
        self._label = label
        self._data = pandas.read_csv(data_path, delimiter=separator, index_col=False)
        self._label_encoder = label_encoder
        self._phrases = list(self.data[phrase])
        self._labels = list(self.data[label])
        self._labels = label_encoder.transform(self.labels)

        self._word1 = [phrase.split(" ")[0] for phrase in self.phrases]
        self._word2 = [phrase.split(" ")[1] for phrase in self.phrases]

        self._samples = []

    @abstractmethod
    def lookup_embedding(self, words):
        return

    @abstractmethod
    def populate_samples(self):
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def phrases(self):
        return self._phrases

    @property
    def labels(self):
        return self._labels

    @property
    def word1(self):
        return self._word1

    @property
    def word2(self):
        return self._word2

    @property
    def samples(self):
        return self._samples

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def label(self):
        return self._label


class SimplePhraseContextualizedDataset(SimplePhraseDataset):
    """
    This class defines a specific dataset used to classify two-word phrases. It expects the csv dataset to have
    a column containing a sentence, a column containing the phrase (first and second word separated with white space),
    a column containing a label. The names of the corresponding columns can be specified.
    """

    def __init__(self, data_path, label_encoder, bert_model,
                 max_len, lower_case, batch_size, separator, phrase, label, context):
        """

        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param bert_model: [String] The Bert model to be used for extracting the contextualized embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether the tokenizer should lower case words or not
        :param separator: [String] the csv separator
        :param phrase: [String] the label of the column the phrase is stored in
        :param label: [String] the label of the column the class label is stored in
        :param context: [String] the label of the column the context sentence is stored in
        """
        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                                batch_size=batch_size)
        super(SimplePhraseContextualizedDataset, self).__init__(data_path, label_encoder, label=label, phrase=phrase,
                                                                separator=separator)
        self._sentences = list(self.data[context])
        self._samples = self.populate_samples()

    def lookup_embedding(self, words):
        return self.feature_extractor.get_single_word_representations(sentences=self.sentences, target_words=words)

    def populate_samples(self):
        word1_embeddings = self.lookup_embedding(self.word1)
        word2_embeddings = self.lookup_embedding(self.word2)
        return [{"w1": word1_embeddings[i], "w2": word2_embeddings[i], "l": self.labels[i]} for i in
                range(len(self.labels))]

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def sentences(self):
        return self._sentences


class SimplePhraseStaticDataset(SimplePhraseDataset):
    """
    This class defines a specific dataset used to classify two-word phrases. It expects the csv dataset to have
    a column containing a sentence, a column containing the phrase (first and second word separated with white
    space),
    a column containing a label. The names of the corresponding columns can be specified.
    """

    def __init__(self, data_path, label_encoder, embedding_path, separator, phrase, label):
        """

        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param embedding_path: [String] the path to the pretrained embeddings
        :param separator: [String] the csv separator
        :param phrase: [String] the label of the column the phrase is stored in
        :param label: [String]the label of the column the class label is stored in
        """
        self._feature_extractor = StaticEmbeddingExtractor(path_to_embeddings=embedding_path)
        super(SimplePhraseStaticDataset, self).__init__(data_path, label_encoder, label=label, phrase=phrase,
                                                        separator=separator)
        self._samples = self.populate_samples()

    def lookup_embedding(self, words):
        return self.feature_extractor.get_array_embeddings(array_words=words)

    def populate_samples(self):
        word1_embeddings = self.lookup_embedding(self.word1)
        word2_embeddings = self.lookup_embedding(self.word2)

        return [{"w1": word1_embeddings[i], "w2": word2_embeddings[i], "l": self.labels[i]} for i in
                range(len(self.labels))]

    @property
    def feature_extractor(self):
        return self._feature_extractor


class PhraseAndContextDatasetStatic(SimplePhraseDataset):
    """
    This class defines a specific dataset used to classify two-word phrases with additional context or a longer
    phrase. It expects the csv dataset to have
    a column containing a sentence, a column containing the phrase (first and second word separated with white
    space), a column containing a label. The names of the corresponding columns can be specified.
    """

    def __init__(self, data_path, label_encoder, embedding_path, tokenizer_model, separator, phrase, context, label):
        """
        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param embedding_path: [String] the path to the pretrained embeddings
        :param tokenizer_model: [String] Uses the Somajo tokenizer for tokenization. Defines the Tokenizer model that
        should be used.
        :param separator: [String] the csv separator (Default = tab)
        :param phrase: [String] the label of the column the phrase is stored in
        :param context: [String]the label of the column the sentences
        :param label: [String]the label of the column the class label is stored in
        """
        self._feature_extractor = StaticEmbeddingExtractor(path_to_embeddings=embedding_path)
        super(PhraseAndContextDatasetStatic, self).__init__(data_path, label_encoder, label=label, phrase=phrase,
                                                            separator=separator)
        self._sentences = list(self.data[context])
        self._tokenizer = somajo.SoMaJo(tokenizer_model, split_camel_case=True, split_sentences=False)
        self._sentences = self.tokenizer.tokenize_text(self.sentences)
        self._sentences = [[token.text for token in sent] for sent in self.sentences]
        self._samples = self.populate_samples()

    def lookup_embedding(self, words):
        return self.feature_extractor.get_array_embeddings(array_words=words)

    def populate_samples(self):
        """
        Looks up the embedding for each word in the phrase. For each sentence it looks up the word embeddings an creates
        a tensor with the corresponding word embeddings. This tensor is then padded to the length of the longest
        sentence
        in the dataset. So the tensor for each sentence contains the same number of word embeddings, padded with
        zero embeddings at the end.
        """
        word1_embeddings = self.lookup_embedding(self.word1)
        word2_embeddings = self.lookup_embedding(self.word2)
        # a list of torch tensors, each torch tensor with size seqlen x emb dim
        sequences = [torch.tensor(self.lookup_embedding(sent)).float() for sent in
                     self.sentences]
        # pad all sequences such that they have embeddings wit 0.0 at the end, the max len is equal to the longest seq
        sequences = nn.utils.rnn.pad_sequence(batch_first=True, sequences=sequences, padding_value=0.0)

        sequence_lengths = np.array([len(words) for words in self.sentences])
        return [{"w1": word1_embeddings[i],
                 "w2": word2_embeddings[i],
                 "seq": sequences[i],
                 "seq_lengths": sequence_lengths[i],
                 "l": self.labels[i]} for i in range(len(self.labels))]

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def sentences(self):
        return self._sentences

    @property
    def tokenizer(self):
        return self._tokenizer


class PhraseAndContextDatasetBert(SimplePhraseDataset):
    """
    This class defines a specific dataset used to classify two-word phrases with additional context or a longer
    phrase. It expects the csv dataset to have
    a column containing a sentence, a column containing the phrase (first and second word separated with white
    space), a column containing a label. The names of the corresponding columns can be specified. It uses contextualized
    Bert embeddings.
    """

    def __init__(self, data_path, label_encoder, bert_model,
                 max_len, lower_case, batch_size, tokenizer_model, separator, phrase, context, label):
        """
        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param bert_model: [String] The Bert model to be used for extracting the contextualized embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether the tokenizer should lower case words or not
        :param separator: [String] the csv separator
        :param tokenizer_model: [String] Uses the Somajo tokenizer for tokenization. Defines the Tokenizer model that
        should be used.
        :param phrase: [String] the label of the column the phrase is stored in
        :param label: [String] the label of the column the class label is stored in
        :param context: [String] the label of the column the context sentence is stored in
        """
        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                                batch_size=batch_size)
        super(PhraseAndContextDatasetBert, self).__init__(data_path, label_encoder, label=label, phrase=phrase,
                                                          separator=separator)
        self._sentences = list(self.data[context])
        self._tokenizer = somajo.SoMaJo(tokenizer_model, split_camel_case=True, split_sentences=False)
        self._sentences = self.tokenizer.tokenize_text(self.sentences)
        self._sentences = [[token.text for token in sent] for sent in self.sentences]
        self._samples = self.populate_samples()

    def lookup_embedding(self, words):
        return self.feature_extractor.get_single_word_representations(sentences=self.sentences, target_words=words)

    def lookup_sequence(self, sentences, words):
        return self.feature_extractor.get_single_word_representations(sentences=sentences, target_words=words)

    def populate_samples(self):
        """
        Looks up the embedding for each word in the phrase. For each sentence it looks up the word embeddings an creates
        a tensor with the corresponding word embeddings. This tensor is then padded to the length of the longest
        sentence
        in the dataset. So the tensor for each sentence contains the same number of word embeddings, padded with
        zero embeddings at the end.
        """
        word1_embeddings = self.lookup_embedding(self.word1)
        word2_embeddings = self.lookup_embedding(self.word2)
        # a list of torch tensors, each torch tensor with size seqlen x emb dim
        sequences = []
        for sent in self.sentences:
            string_sent = " ".join(sent)
            context_sents = len(sent) * [string_sent]
            sequences.append(torch.tensor(self.lookup_sequence(context_sents, sent)).float())

        # pad all sequences such that they have embeddings wit 0.0 at the end, the max len is equal to the longest seq
        sequences = nn.utils.rnn.pad_sequence(batch_first=True, sequences=sequences, padding_value=0.0)

        sequence_lengths = np.array([len(words) for words in self.sentences])

        return [{"w1": word1_embeddings[i],
                 "w2": word2_embeddings[i],
                 "seq": sequences[i],
                 "seq_lengths": sequence_lengths[i],
                 "l": self.labels[i]} for i in range(len(self.labels))]

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def sentences(self):
        return self._sentences

    @property
    def tokenizer(self):
        return self._tokenizer


class StaticRankingDataset(Dataset):

    def __init__(self, data_path, embedding_path, separator, mod, head, phrase):
        """
        This datasets can be used to pretrain a composition model on a reconstruction task
        :param data_path: the path to the dataset, should have a header
        :param embedding_path: the path to the pretrained static word embeddings
        :param separator: the separator within the dataset (default = whitespace)
        :param mod: the name of the column holding the modifier words
        :param head: the name of the column holding the head words
        :param phrase: the name of the column holding the phrases
        """
        self._data = pandas.read_csv(data_path, delimiter=separator, index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._phrases = list(self.data[phrase])
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.phrases), "invalid input data, different lenghts"

        self._feature_extractor = StaticEmbeddingExtractor(path_to_embeddings=embedding_path)
        self._samples = self.populate_samples()

    def lookup_embedding(self, words):
        return self.feature_extractor.get_array_embeddings(array_words=words)

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and phrases and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and phrase embeddings (w1, w2, l)
        """
        word1_embeddings = self.lookup_embedding(self.modifier_words)
        word2_embeddings = self.lookup_embedding(self.head_words)
        label_embeddings = self.lookup_embedding(self.phrases)
        return [
            {"w1": word1_embeddings[i], "w2": word2_embeddings[i], "l": label_embeddings[i], "phrase": self.phrases[i]}
            for i in
            range(len(self.phrases))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def phrases(self):
        return self._phrases

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def samples(self):
        return self._samples


class MultiRankingDataset(Dataset):
    """
    This dataset can be used to combine two different dataset into one. The datasets need to be of the same type.
    """

    def __init__(self, dataset_1, dataset_2):
        assert type(dataset_1) == type(dataset_2), "cannot combine two datasets of different types"
        self._dataset_1 = dataset_1
        self._dataset_2 = dataset_2

    def __len__(self):
        if len(self.dataset_2) < len(self.dataset_1):
            return len(self.dataset_2)
        return len(self.dataset_1)

    def __getitem__(self, idx):
        """Returns a batch for each dataset"""
        task1 = self.dataset_1[idx]
        task2 = self.dataset_2[idx]
        return task1, task2

    @property
    def dataset_1(self):
        return self._dataset_1

    @property
    def dataset_2(self):
        return self._dataset_2


class ContextualizedRankingDataset(Dataset):

    def __init__(self, data_path, bert_model, max_len, lower_case, batch_size, separator, mod, head,
                 label, label_definition_path):
        """
        This Dataset can be used to train a composition model with contextualized embeddings to create attribute-like
        representations
        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param bert_model: [String] The Bert model to be used for extracting the contextualized embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether the tokenizer should lower case words or not
        :param separator: [String] the csv separator
        :param label: [String] the label of the column the class label is stored in
        :param mod: [String] the label of the column the modifier is stored in
        :param head: [String] the label of the column the head is stored in
        :param label_definition_path: [String] path to the file that holds the definitions for the labels
        """
        self._data = pandas.read_csv(data_path, delimiter=separator, index_col=False)
        self._definitions = pandas.read_csv(label_definition_path, delimiter="\t", index_col=False)
        self._modifier_words = list(self.data[mod])
        self._head_words = list(self.data[head])
        self._phrases = [self.modifier_words[i] + " " + self.head_words[i] for i in range(len(self.data))]
        self._labels = list(self.data[label])
        self._label2definition = dict(zip(list(self._definitions["label"]), list(self._definitions["definition"])))
        self._label_definitions = [self._label2definition[l] for l in self.labels]
        assert len(self.modifier_words) == len(self.head_words) == len(
            self.phrases), "invalid input data, different lenghts"

        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                                batch_size=batch_size)
        self._samples = self.populate_samples()

    def lookup_embedding(self, simple_phrases, target_words):
        return self.feature_extractor.get_single_word_representations(target_words=target_words,
                                                                      sentences=simple_phrases)

    def populate_samples(self):
        """
        Looks up the embeddings for all modifier, heads and labels and stores them in a dictionary
        :return: List of dictionary objects, each storing the modifier, head and phrase embeddings (w1, w2, l)
        """
        word1_embeddings = self.lookup_embedding(target_words=self.modifier_words, simple_phrases=self.phrases)
        word2_embeddings = self.lookup_embedding(target_words=self.head_words, simple_phrases=self.phrases)
        label_embeddings = self.lookup_embedding(target_words=self.labels, simple_phrases=self._label_definitions)
        word1_embeddings = F.normalize(word1_embeddings, p=2, dim=1)
        word2_embeddings = F.normalize(word2_embeddings, p=2, dim=1)
        label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
        return [
            {"w1": word1_embeddings[i], "w2": word2_embeddings[i], "l": label_embeddings[i], "label": self.labels[i]}
            for i in
            range(len(self.phrases))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def data(self):
        return self._data

    @property
    def modifier_words(self):
        return self._modifier_words

    @property
    def head_words(self):
        return self._head_words

    @property
    def phrases(self):
        return self._phrases

    @property
    def labels(self):
        return self._labels

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def samples(self):
        return self._samples

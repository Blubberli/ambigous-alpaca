from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas
import torch
from torch import nn
import somajo
import numpy as np
from scripts import BertExtractor, StaticEmbeddingExtractor


class SimplePhraseDataset(ABC, Dataset):
    def __init__(self, data_path, separator="\t", phrase="phrase", label="label"):
        self._data = pandas.read_csv(data_path, delimiter=separator, index_col=False)
        self._phrases = list(self.data[phrase])
        self._labels = list(self.data[label])
        self._label_encoder = LabelEncoder()
        self._labels, self._label2index = self.encode_labels()

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

    def encode_labels(self):
        values = self.label_encoder.fit_transform(self.labels)
        keys = self.label_encoder.classes_
        label2index = dict(zip(keys, values))
        return self.label_encoder.fit_transform(self.labels), label2index

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
    def label2index(self):
        return self._label2index


class SimplePhraseContextualizedDataset(SimplePhraseDataset):
    """
    This class defines a specific dataset used to classify two-word phrases. It expects the csv dataset to have
    a column containing a sentence, a column containing the phrase (first and second word separated with white space),
    a column containing a label. The names of the corresponding columns can be specified.
    """

    def __init__(self, data_path, bert_model,
                 max_len, lower_case, batch_size, separator="\t", phrase="phrase", label="label", context="context"):
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
        super(SimplePhraseContextualizedDataset, self).__init__(data_path)
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

    def __init__(self, data_path, embedding_path, separator="\t", phrase="phrase", label="label"):
        """

        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param embedding_path: [String] the path to the pretrained embeddings
        :param separator: [String] the csv separator
        :param phrase: [String] the label of the column the phrase is stored in
        :param label: [String]the label of the column the class label is stored in
        """
        self._feature_extractor = StaticEmbeddingExtractor(path_to_embeddings=embedding_path)
        super(SimplePhraseStaticDataset, self).__init__(data_path)
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

    def __init__(self, data_path, embedding_path, tokenizer_model, separator="\t", phrase="phrase", context="context",
                 label="label"):
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
        super(PhraseAndContextDatasetStatic, self).__init__(data_path)
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
        a tensor with the corresponding word embeddings. This tensor is then padded to the length of the longest sentence
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

    def __init__(self, data_path, bert_model,
                 max_len, lower_case, batch_size, tokenizer_model, separator="\t", phrase="phrase", context="context",
                 label="label"):
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
        super(PhraseAndContextDatasetBert, self).__init__(data_path)
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
        a tensor with the corresponding word embeddings. This tensor is then padded to the length of the longest sentence
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

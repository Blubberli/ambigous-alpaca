from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas
from scripts import BertExtractor


class SimplePhraseDataset(ABC, Dataset):
    def __init__(self, data_path, separator="\t", phrase="phrase", label="label"):
        self._data = pandas.read_csv(data_path, delimiter=separator)
        self._phrases = self.data[phrase]
        self._labels = self.data[label]
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
        keys = self.label_encoder.classes_
        values = self.label_encoder.transform(keys)
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
                 max_len, lower_case, separator="\t", phrase="phrase", label="label", context="context"):
        """

        :param data_path: [String] The path to the csv datafile that needs to be transformed into a dataset.
        :param bert_model: [String] The Bert model to be used for extracting the contextualized embeddings
        :param max_len: [int] the maximum length of word pieces (can be a large number)
        :param lower_case: [boolean] Whether 
        :param separator:
        :param phrase:
        :param label:
        :param context:
        """
        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case)
        super(SimplePhraseContextualizedDataset, self).__init__(data_path)
        self._sentences = self.data[context]
        self._samples = self.populate_samples()

    def lookup_embedding(self, words):
        return self.feature_extractor.get_single_word_representations(sentences=self.sentences, target_words=words)

    def populate_samples(self):
        word1_embeddings = self.lookup_embedding(self.word1)
        word2_embeddings = self.lookup_embedding(self.word2)
        return [[word1_embeddings[i], word2_embeddings[i], self.labels[i]] for i in range(len(self.labels))]

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def sentences(self):
        return self._sentences

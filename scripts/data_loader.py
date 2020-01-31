from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import pandas
from scripts import BertExtractor


class SimplePhraseDataset(ABC, Dataset):
    def __init__(self, data_path, separator="\t", phrase="phrase", label="label"):
        self._data = pandas.read_csv(data_path, delimiter=separator)
        self._phrases = self.data[phrase]
        self._labels = self.data[label]
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


class SimplePhraseContextualizedDataset(SimplePhraseDataset):
    def __init__(self, data_path, bert_model,
                 max_len, lower_case, separator="\t", phrase="phrase", label="label", context="context"):
        super(SimplePhraseContextualizedDataset, self).__init__(data_path)
        self._feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case)
        self._sentences = self.data[context]
        self._samples = self.populate_samples()

    def lookup_embedding(self, words):
        return self.feature_extractor.get_single_word_representations(sentences=self.sentences, target_words=words)

    def populate_samples(self):
        word1_embeddings = self.lookup_embedding(self.word1)
        word2_embeddings = self.lookup_embedding(self.word2)
        return [word1_embeddings, word2_embeddings, self.labels]

    @property
    def feature_extractor(self):
        return self.feature_extractor

    @property
    def sentences(self):
        return self._sentences



if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data = "/home/neele/PycharmProjects/ambigous-alpaca/tests/data_multiclassification/test.txt"
    dataset = SimplePhraseContextualizedDataset(data, 'bert-base-german-cased', 20, False)
    print(dataset[10])
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        print(i, batch)

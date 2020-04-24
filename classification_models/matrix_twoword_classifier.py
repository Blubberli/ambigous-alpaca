import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.composition_functions as comp_functions


class MatrixTwoWordClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate, normalize_embeddings,
                 add_single_words=False):
        """
        this class includes a two-word classifier with one hidden layer and one output layer.
        the classifier first combines the two words with a matrix that will be trained during training and then uses
        the
        composed representation and eventually the single word representations as input to the classification layer
        :param input_dim: embedding size
        :param hidden_dim: dimension of hidden layer of classifier
        :param label_nr: number of classes to predict
        :param dropout_rate: dropout rate for regularization
        :param normalize_embeddings: whether the composed representation should be normalized to unit length
        :param add_single_words: if True, the classifier will get a concatenation of the single word embeddings with
        the composed representation, otherwise the composed representation only
        """
        super(MatrixTwoWordClassifier, self).__init__()
        self._matrix_layer = nn.Linear(input_dim * 2, input_dim)
        forward_dim = input_dim
        self._add_single_words = add_single_words
        self._normalize_embeddings = normalize_embeddings
        if self.add_single_words:
            forward_dim = input_dim * 3
        self._hidden_layer = nn.Linear(forward_dim, hidden_dim)
        self._output_layer = nn.Linear(hidden_dim, label_nr)
        self._dropout_rate = dropout_rate

    def forward(self, batch):
        """
        this function takes two words, combines them via the matrix composition function send them through a
        nonlinear forward layer, eventually after concatenating the single word embeddings.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the raw label scores
        """
        device = batch["device"]
        self._composed_phrase = self.compose(batch["w1"].to(device), batch["w2"].to(device))
        if self.add_single_words:
            w1_w2 = torch.cat((batch["w1"].to(device), batch["w2"].to(device)), 1)
            self._composed_phrase = torch.cat((w1_w2, self.composed_phrase), 1)
        x = F.relu(self.hidden_layer(self.composed_phrase))
        x = F.dropout(x, p=self.dropout_rate)
        return self.output_layer(x)

    def compose(self, word1, word2):
        """
        Two words are combined via a matrix : relu(word1;word2 x W)+b
        :param word1: embedding of the first word
        :param word2: embedding of the second word
        :return: composed input word embeddings (dimension = embedding dimension)
        """
        composed_phrase = comp_functions.concat(word1, word2, axis=1)
        transformed = self.matrix_layer(composed_phrase)
        reg_transformed = F.dropout(transformed, p=self.dropout_rate)
        if self.normalize_embeddings:
            reg_transformed = F.normalize(reg_transformed, p=2, dim=1)
        return reg_transformed

    @property
    def hidden_layer(self):
        return self._hidden_layer

    @property
    def output_layer(self):
        return self._output_layer

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def matrix_layer(self):
        return self._matrix_layer

    @property
    def normalize_embeddings(self):
        return self._normalize_embeddings

    @property
    def composed_phrase(self):
        return self._composed_phrase

    @property
    def add_single_words(self):
        return self._add_single_words

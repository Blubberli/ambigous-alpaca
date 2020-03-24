import torch.nn as nn
import torch.nn.functional as F
import scripts.composition_functions as comp_functions


class MatrixPretrain(nn.Module):

    def __init__(self, input_dim, dropout_rate, normalize_embeddings):
        """
        This class cotanins the matrix composition model that can be used to train a model with the cosine distance loss
        The model takes two words and returns their composed representation.
        :param input_dim: embedding dimension
        :param dropout_rate: dropout rate for regularization
        :param normalize_embeddings: whether the composed representation should be normalized to unit length
        """
        super(MatrixPretrain, self).__init__()
        self._matrix_layer = nn.Linear(input_dim * 2, input_dim)
        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, batch):
        """
        this function takes two words and combines them via the matrix composition model.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the composed representation
        """
        device = batch["device"]
        self._composed_phrase = self.compose(batch["w1"].to(device), batch["w2"].to(device))
        return self.composed_phrase

    def compose(self, word1, word2):
        """
        this function takes two words, concatenates them and applies a non-linear matrix transformation (
        hidden layer)
        Its output is then fed to an output layer. Then it returns the concatenated and transformed vectors.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the raw label scores
        """
        composed_phrase = comp_functions.concat(word1, word2, axis=1)
        transformed = F.relu(self.matrix_layer(composed_phrase))
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

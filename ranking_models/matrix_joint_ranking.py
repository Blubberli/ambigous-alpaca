import torch.nn as nn
import torch.nn.functional as F
import utils.composition_functions as comp_functions


class MatrixJointRanker(nn.Module):
    """
    This model trains two representations, each containing a different type of semantic content (one for the general
    phrase and one to be close the the conveyed attribute). Both representations are combined into one representation.
    The model is a basic matrix composition model with two specific linear transformation for each type of semantic representation.
    """

    def __init__(self, input_dim, dropout_rate, normalize_embeddings):
        """
        :param input_dim: embedding dimension
        :param dropout_rate: dropout rate for regularization
        :param normalize_embeddings: whether the composed representation should be normalized to unit length
        :param rep1_weight: weight for representation 1 for final representation
        :param rep2_weight: weight for representation 2 for final representation
        """
        super(MatrixJointRanker, self).__init__()
        self._matrix_layer = nn.Linear(input_dim * 2, input_dim)
        self._specific_matrix_1 = nn.Linear(input_dim, input_dim)
        self._specific_matrix_2 = nn.Linear(input_dim, input_dim)
        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, batch):
        """
        Takes a data batch as input that contains two word vectors. It applies the same matrix to compose the two vectors.
        on top of that, two further linear transformations are applied to generate specific representations: one that should
        look like the phrase and one that should resemble the attribute.
        :param batch: a dictionary
        :return: the final composed phrase, representation 1, representation 2
        """
        device = batch["device"]
        self._composed_phrase = self.compose(batch["w1"].to(device), batch["w2"].to(device))
        self._representation_1 = self._specific_matrix_1(self._composed_phrase)
        self._representation_2 = self._specific_matrix_2(self._composed_phrase)
        if self.normalize_embeddings:
            self._representation_1 = F.normalize(self._representation_1, p=2, dim=1)
            self._representation_2 = F.normalize(self._representation_2, p=2, dim=1)

        self._final_composed_phrase = self.representation_1 + self.representation_2

        if self.normalize_embeddings:
            self._final_composed_phrase = F.normalize(self._final_composed_phrase, p=2, dim=1)
            self._representation_1 = F.normalize(self._representation_1, p=2, dim=1)
            self._representation_2 = F.normalize(self._representation_2, p=2, dim=1)

        return self.final_composed_phrase, self.representation_1, self.representation_2

    def compose(self, word1, word2):
        """
        this function takes two words, concatenates them and applies linear transformation
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the composed representation
        """
        composed_phrase = comp_functions.concat(word1, word2, axis=1)
        transformed = self.matrix_layer(composed_phrase)
        reg_transformed = F.dropout(transformed, p=self.dropout_rate)
        if self.normalize_embeddings:
            reg_transformed = F.normalize(reg_transformed, p=2, dim=1)
        return reg_transformed

    @property
    def matrix_layer(self):
        return self._matrix_layer

    @property
    def specific_matrix1(self):
        return self._specific_matrix_1

    @property
    def specific_matrix2(self):
        return self._specific_matrix_2

    @property
    def final_composed_phrase(self):
        return self._final_composed_phrase

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def normalize_embeddings(self):
        return self._normalize_embeddings

    @property
    def composed_phrase(self):
        return self._composed_phrase

    @property
    def representation_1(self):
        return self._representation_1

    @property
    def representation_2(self):
        return self._representation_2


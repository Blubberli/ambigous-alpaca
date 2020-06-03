import torch.nn as nn
import torch
import torch.nn.functional as F
import utils.composition_functions as comp_functions


class FullAdditive(nn.Module):

    def __init__(self, input_dim, dropout_rate, normalize_embeddings):
        """
        This class contains the full additive composition model that can be used to train a model with the cosine
        distance loss
        The model takes two words and returns their composed representation.
        :param input_dim: embedding dimension
        :param dropout_rate: dropout rate for regularization
        :param normalize_embeddings: whether the composed representation should be normalized to unit length
        """
        super(FullAdditive, self).__init__()
        self._adj_matrix = nn.Parameter(torch.eye(input_dim), requires_grad=True)
        self._noun_matrix = nn.Parameter(torch.eye(input_dim), requires_grad=True)

        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, batch):
        """
        this function takes two words and combines them via the full additive composition model.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the composed representation
        """
        device = batch["device"]
        self._composed_phrase = self.compose(batch["w1"].to(device), batch["w2"].to(device))
        return self.composed_phrase

    def compose(self, word1, word2):
        """
        this function takes two words, each will be transformed by a separate matrix. the transformed vectors are summed
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the composed phrase
        """
        composed_phrase = comp_functions.full_additive(modifier_matrix=self.adj_matrix, modifier_vector=word1,
                                                       head_matrix=self.noun_matrix, head_vector=word2)
        if self.normalize_embeddings:
            composed_phrase = F.normalize(composed_phrase, p=2, dim=1)
        return composed_phrase

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
    def adj_matrix(self):
        return self._adj_matrix

    @property
    def noun_matrix(self):
        return self._noun_matrix

    @property
    def normalize_embeddings(self):
        return self._normalize_embeddings

    @property
    def composed_phrase(self):
        return self._composed_phrase

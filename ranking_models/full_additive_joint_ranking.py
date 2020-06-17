import torch.nn as nn
import torch.nn.functional as F
import utils.composition_functions as comp_functions
import torch


class FullAdditiveJointRanker(nn.Module):
    """
    This class contains the full additive composition model JOINT model. it trains two representations, each containing
    a different type of semantic content (one for the general phrase and one to be close the the conveyed attribute).
    Both representations are combined into one representation.
    The model takes two words and returns their composed representation.
    :param input_dim: embedding dimension
    :param dropout_rate: dropout rate for regularization
    :param normalize_embeddings: whether the composed representation should be normalized to unit length
    """

    def __init__(self, input_dim, normalize_embeddings, non_linearity):
        """
        :param input_dim: embedding dimension
        :param dropout_rate: dropout rate for regularization
        :param normalize_embeddings: whether the composed representation should be normalized to unit length
        """
        super(FullAdditiveJointRanker, self).__init__()

        self._general_adj_matrix = nn.Parameter(torch.eye(input_dim), requires_grad=True)
        self._general_noun_matrix = nn.Parameter(torch.eye(input_dim), requires_grad=True)

        self._adj_matrix_1 = nn.Parameter(torch.eye(input_dim), requires_grad=True)
        self._noun_matrix_1 = nn.Parameter(torch.eye(input_dim), requires_grad=True)

        self._adj_matrix_2 = nn.Parameter(torch.eye(input_dim), requires_grad=True)
        self._noun_matrix_2 = nn.Parameter(torch.eye(input_dim), requires_grad=True)

        self._normalize_embeddings = normalize_embeddings

        self._non_linearity = non_linearity

    def forward(self, batch):
        """
        this function takes two words, applies a general matrix transofrmation to first and second word and then
        applies linear transformations with task specific adjective and noun matrices . The transformed vectors are combined
        and these combined vectors are again combined to form the combined output vector
        :param batch: a dictionary
        :return: the final composed phrase, representation 1, representation 2
        """
        device = batch["device"]

        word_1 = batch["w1"].to(device)
        word_2 = batch["w2"].to(device)

        # general transformation
        adj_vector = word_1.matmul(self._general_adj_matrix)
        noun_vector = word_2.matmul(self._general_noun_matrix)
        if self._non_linearity:
            adj_vector = F.relu(adj_vector)
            noun_vector = F.relu(noun_vector)
        if self.normalize_embeddings:
            adj_vector = F.normalize(adj_vector, p=2, dim=1)
            noun_vector = F.normalize(noun_vector, p=2, dim=1)

        self._composed_phrase_1, self._composed_phrase_2 = self.compose(adj_vector, noun_vector)

        self._final_composed_phrase = self._composed_phrase_1 + self._composed_phrase_2
        if self.normalize_embeddings:
            self._composed_phrase_1 = F.normalize(self._composed_phrase_1, p=2, dim=1)
            self._composed_phrase_2 = F.normalize(self._composed_phrase_2, p=2, dim=1)
            self._final_composed_phrase = F.normalize(self._final_composed_phrase, p=2, dim=1)
        return self.final_composed_phrase, self.composed_phrase_1, self.composed_phrase_2

    def compose(self, adj_vector, noun_vector):
        """
        this function takes two words, each will be transformed twice by a separate matrix.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: both composed phrases
        """
        composed_phrase_1 = comp_functions.full_additive(modifier_matrix=self._adj_matrix_1, modifier_vector=adj_vector,
                                                         head_matrix=self._noun_matrix_1, head_vector=noun_vector)
        composed_phrase_2 = comp_functions.full_additive(modifier_matrix=self.adj_matrix_2, modifier_vector=adj_vector,
                                                         head_matrix=self._noun_matrix_2, head_vector=noun_vector)

        return composed_phrase_1, composed_phrase_2

    @property
    def composed_phrase_1(self):
        return self._composed_phrase_1

    @property
    def composed_phrase_2(self):
        return self._composed_phrase_2

    @property
    def general_adj_matrix(self):
        return self._general_adj_matrix

    @property
    def general_noun_matrix(self):
        return self._general_noun_matrix

    @property
    def adj_matrix_1(self):
        return self._adj_matrix_1

    def noun_matrix_1(self):
        return self._noun_matrix_1

    @property
    def adj_matrix_2(self):
        return self._adj_matrix_2

    @property
    def noun_matrix_2(self):
        return self._adj_matrix_2

    @property
    def final_composed_phrase(self):
        return self._final_composed_phrase

    @property
    def normalize_embeddings(self):
        return self._normalize_embeddings

    @property
    def non_linearity(self):
        return self._non_linearity

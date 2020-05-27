import torch
from torch import nn
from torch.functional import F
from utils.composition_functions import transweigh


class TransweighJointRanker(nn.Module):
    """
    This model trains two representations, each containing a different type of semantic content (one for the general
    phrase and one to be close the the conveyed attribute.
    """

    def __init__(self, input_dim, dropout_rate, transformations, normalize_embeddings):
        super().__init__()

        # the transformation tensor and bias for transforming the input vectors
        self._transformation_tensor = nn.Parameter(
            torch.empty(transformations, 2 * input_dim, input_dim), requires_grad=True)
        nn.init.xavier_normal_(self.transformation_tensor)
        self._transformation_bias = nn.Parameter(torch.empty(transformations, input_dim), requires_grad=True)
        nn.init.uniform_(self.transformation_bias)

        # - the combining tensor for representation 1
        self._combining_tensor_1 = nn.Parameter(data=torch.empty(transformations, input_dim, input_dim),
                                                requires_grad=True)
        nn.init.xavier_normal_(self.combining_tensor_1)
        self._combining_bias_1 = nn.Parameter(torch.empty(input_dim), requires_grad=True)
        nn.init.uniform_(self.combining_bias_1)

        # - the combining tensor for representation 2
        self._combining_tensor_2 = nn.Parameter(data=torch.empty(transformations, input_dim, input_dim),
                                                requires_grad=True)
        nn.init.xavier_normal_(self.combining_tensor_2)
        self._combining_bias_2 = nn.Parameter(torch.empty(input_dim), requires_grad=True)
        nn.init.uniform_(self.combining_bias_2)

        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, batch):
        """
        Takes a data batch as input that should contain two word vectors. They are composed two times to get two
        representations. Both composition functions share the transformations but for each representation a different
        weighting is applied. A final representation is constructed using a weighted summation of both representations.
        :param batch: a dictionary
        :return: the final composed phrase, representation 1, representation 2
        """
        device = batch["device"]
        self._representation_1 = self.compose(word1=batch["w1"].to(device), word2=batch["w2"].to(device),
                                              combinig_tensor=self.combining_tensor_1,
                                              combining_bias=self.combining_bias_1)
        self._representation_2 = self.compose(word1=batch["w1"].to(device), word2=batch["w2"].to(device),
                                              combinig_tensor=self.combining_tensor_2,
                                              combining_bias=self.combining_bias_2)
        self._composed_phrase = self.representation_1 + self.representation_2
        if self.normalize_embeddings:
            self._composed_phrase = F.normalize(self.composed_phrase, p=2, dim=1)
        return self.composed_phrase, self.representation_1, self.representation_2

    def compose(self, word1, word2, combinig_tensor, combining_bias):
        """
        This functions composes two input representations with the transformation weighting model. If set to True,
        the composed representation is normalized
        :param word1: the representation of the first word (torch tensor)
        :param word2: the representation of the second word (torch tensor)
        :param combinig_tensor: The tensor used for weighting the transformed input vectors into one representation
        :param combining_bias: The corresponding bias
        :return: a composed representation
        """
        composed_phrase = transweigh(word1=word1, word2=word2, transformation_tensor=self.transformation_tensor,
                                     transformation_bias=self.transformation_bias,
                                     combining_bias=combining_bias,
                                     combining_tensor=combinig_tensor, dropout_rate=self.dropout_rate,
                                     training=self.training)
        if self.normalize_embeddings:
            composed_phrase = F.normalize(composed_phrase, p=2, dim=1)
        return composed_phrase

    @property
    def combining_tensor_1(self):
        return self._combining_tensor_1

    @property
    def combining_bias_1(self):
        return self._combining_bias_1

    @property
    def combining_tensor_2(self):
        return self._combining_tensor_2

    @property
    def combining_bias_2(self):
        return self._combining_bias_2

    @property
    def transformation_tensor(self):
        return self._transformation_tensor

    @property
    def transformation_bias(self):
        return self._transformation_bias

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

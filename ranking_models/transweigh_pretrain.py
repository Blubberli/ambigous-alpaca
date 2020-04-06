import torch
from torch import nn
from torch.functional import F
from utils.composition_functions import transweigh


class TransweighPretrain(nn.Module):
    """
    This class contains a model that takes two (batches of) word representations as input and combines them via
    transformations. The transformed, composed vector can be optimized compared to a gold representation.
    The composition model can thus be trained on reconstructing a gold representation of a two-word phrase.
    """

    def __init__(self, input_dim, dropout_rate, transformations, normalize_embeddings):
        super().__init__()
        # --- we create the following variable that will be trainable parameters for our classifier:

        # the transformation tensor for transforming the input vectors
        self._transformation_tensor = nn.Parameter(
            torch.empty(transformations, 2 * input_dim, input_dim), requires_grad=True)
        nn.init.xavier_normal_(self.transformation_tensor)
        self._transformation_bias = nn.Parameter(torch.empty(transformations, input_dim), requires_grad=True)
        nn.init.uniform_(self.transformation_bias)
        # - the combining tensor combines the transformed phrase representation into a final, flat vector
        self._combining_tensor = nn.Parameter(data=torch.empty(transformations, input_dim, input_dim),
                                              requires_grad=True)
        nn.init.xavier_normal_(self.combining_tensor)
        self._combining_bias = nn.Parameter(torch.empty(input_dim), requires_grad=True)
        nn.init.uniform_(self.combining_bias)
        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, batch):
        """
        Composes the input vectors into one representation.
        :param word1: word1: the representation of the first word (torch tensor)
        :param word2: word2: the representation of the second word (torch tensor)
        :return: the composed representation, same shape as each input representation
        """
        self._composed_phrase = self.compose(batch["w1"], batch["w2"])
        return self.composed_phrase

    def compose(self, word1, word2):
        """
        This functions composes two input representations with the transformation weighting model. If set to True,
        the composed representation is normalized
        :param word1: the representation of the first word (torch tensor)
        :param word2: the representation of the second word (torch tensor)
        :param training: True if the model should be trained, False if the model is in inference
        :return: the composed vector representation, eventually normalized to unit norm
        """
        composed_phrase = transweigh(word1=word1, word2=word2, transformation_tensor=self.transformation_tensor,
                                     transformation_bias=self.transformation_bias, combining_bias=self.combining_bias,
                                     combining_tensor=self.combining_tensor, dropout_rate=self.dropout_rate,
                                     training=self.training)
        if self.normalize_embeddings:
            composed_phrase = F.normalize(composed_phrase, p=2, dim=1)
        return composed_phrase

    @property
    def combining_tensor(self):
        return self._combining_tensor

    @property
    def combining_bias(self):
        return self._combining_bias

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

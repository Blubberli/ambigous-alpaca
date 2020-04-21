import torch
from torch import nn
from torch.functional import F
from utils.composition_functions import transweigh


class TransweighTwoWordClassifier(nn.Module):
    """
    This class contains a classifier that takes two (batches of) word representations as input and combines them via
    transformations. The transformed, composed vector is used as input to a hidden layer and an output layer to predict
    a number of classes. The hidden layer and all parameters needed to compose the vectors are trained towards the
    objective
    """

    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate, transformations, normalize_embeddings,
                 add_single_words=False):
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
        # - these variables are needed for the classifier (one transformation, one output layer)s
        self._add_single_words = add_single_words
        if self.add_single_words:
            input_dim = input_dim * 3
        self._hidden = nn.Linear(input_dim, hidden_dim)
        self._output = nn.Linear(self.hidden.out_features, label_nr)
        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, batch):
        """
        First composes the input vectors into one representation. This is then feed trough a hidden layer with a Relu
        and
        finally trough an output layer that returns weights for each class.
        :param word1: word1: the representation of the first word (torch tensor)
        :param word2: word2: the representation of the second word (torch tensor)
        :param training: training: True if the model should be trained, False if the model is in inference
        :return: the raw weights for each class
        """
        device = batch["device"]
        self._composed_phrase = self.compose(batch["w1"].to(device), batch["w2"].to(device), self.training)
        if self.add_single_words:
            w1_w2 = torch.cat((batch["w1"].to(device), batch["w2"].to(device)), 1)
            self._composed_phrase = torch.cat((w1_w2, self.composed_phrase), 1)
        hidden = F.relu(self.hidden(self.composed_phrase))
        hidden = F.dropout(hidden, p=self.dropout_rate)
        class_weights = self.output(hidden)
        return class_weights

    def compose(self, word1, word2, training):
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
                                     training=training)
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
    def hidden(self):
        return self._hidden

    @property
    def output(self):
        return self._output

    @property
    def composed_phrase(self):
        return self._composed_phrase

    @property
    def add_single_words(self):
        return self._add_single_words

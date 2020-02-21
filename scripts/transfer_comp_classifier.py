import torch
from torch import nn
from torch.functional import F
from scripts import transweigh


class TransferCompClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate, normalize_embeddings, pretrained_model):
        """
        initialises the transweigh composition model (the transformation tensor and bias as well as the combining tensor and bias) with weights from saved model
        the weights for the classifier are not loaded from the saved model
        is based on trasnweigh_twoword_classifier
        :param input_dim: size of input dimension
        :param hidden_dim: size of hidden dimension
        :param label_nr: nr of labels to be predicted
        :param dropout_rate: how high dropout rate should be
        :param normalize_embeddings: True if embeddings should be normalized
        :param pretrained_model: path to saved model
        """
        super().__init__()
        self._pretrained_model = torch.load(pretrained_model)

        self._transformation_tensor = nn.Parameter(self._pretrained_model["_transformation_tensor"], requires_grad=True)
        self._transformation_bias = nn.Parameter(self._pretrained_model["_transformation_bias"], requires_grad=True)
        self._combining_tensor = nn.Parameter(self._pretrained_model["_combining_tensor"], requires_grad=True)
        self._combining_bias = nn.Parameter(self._pretrained_model["_combining_bias"], requires_grad=True)
        self._hidden = nn.Linear(input_dim, hidden_dim)
        self._output = nn.Linear(self.hidden.out_features, label_nr)
        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings

    def forward(self, word1, word2, training):
        """
        First composes the input vectors into one representation. This is then feed trough a hidden layer with a Relu and
        finally trough an output layer that returns weights for each class.
        :param word1: word1: the representation of the first word (torch tensor)
        :param word2: word2: the representation of the second word (torch tensor)
        :param training: training: True if the model should be trained, False if the model is in inference
        :return: the raw weights for each class
        """

        self._composed_phrase = self.compose(word1, word2, training)
        hidden = F.relu(self.hidden(self.composed_phrase))
        hidden = F.dropout(hidden, training=training, p=self.dropout_rate)
        class_weights = self.output(hidden)
        return class_weights

    def compose(self, word1, word2, training):
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
    def pretrained_model(self):
        return self._pretrained_model

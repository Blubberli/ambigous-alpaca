import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.composition_functions as comp_functions


class MatrixTransferRanker(nn.Module):

    def __init__(self, dropout_rate, normalize_embeddings, pretrained_model):
        """
        this class combines two words with a matrix that will be trained during training.
        The weight and bias of the composition model are loaded from a pretrained model and then trained further.
        :param dropout_rate: dropout rate for regularization
        :param normalize_embeddings: whether the composed representation should be normalized to unit length
        :param pretrained_model: a pretrained torch model of the type (MatrixTwoWordClassifier)
        """
        super(MatrixTransferRanker, self).__init__()
        self._pretrained_model = torch.load(pretrained_model)
        # init matrix model with pretrained weights
        self._pretrained_w = nn.Parameter(self.pretrained_model['_matrix_layer.weight'], requires_grad=True)
        self._pretrained_b = nn.Parameter(self.pretrained_model['_matrix_layer.bias'], requires_grad=True)
        self._matrix_layer = nn.Linear(self.pretrained_w.shape[0], self.pretrained_w.shape[1])
        self._matrix_layer.weight = self._pretrained_w
        self._matrix_layer.bias = self._pretrained_b

        self._normalize_embeddings = normalize_embeddings
        self._dropout_rate = dropout_rate

    def forward(self, batch):
        """
        this function takes two words, concatenates them and applies a non-linear matrix transformation (hidden layer)
        Its output is this transformed, composed representation.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the composed representation
        """
        device = batch["device"]
        self._composed_phrase = self.compose(batch["w1"].to(device), batch["w2"].to(device))
        return self.composed_phrase

    def compose(self, word1, word2):
        """
        Two words are combined via a matrix : word1;word2 x W+b
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
    def pretrained_w(self):
        return self._pretrained_w

    @property
    def pretrained_b(self):
        return self._pretrained_b

    @property
    def pretrained_model(self):
        return self._pretrained_model

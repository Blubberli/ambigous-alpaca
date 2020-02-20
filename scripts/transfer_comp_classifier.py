import torch
from torch import nn
from torch.functional import F
from scripts import transweigh


class TransferCompClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate, normalize_embeddings, pretrained_model):
        super().__init__()
        self._model = torch.load(pretrained_model)
        self._transformation_tensor = self._model["_transformation_tensor"]
        self._transformation_bias = self._model["_transformation_bias"]
        self._combining_tensor = self._model["_combining_tensor"]
        self._combining_bias = self._model["_combining_bias"]
        self._hidden = nn.Linear(input_dim, hidden_dim)
        self._output = nn.Linear(self.hidden.out_features, label_nr)
        self._dropout_rate = dropout_rate
        self._normalize_embeddings = normalize_embeddings


    def forward(self,word1, word2, training):

        self._composed_phrase = self.compose(word1, word2, training)
        hidden = F.relu(self.hidden(self.composed_phrase))
        hidden = F.dropout(hidden, training=training, p=self.dropout_rate)
        class_weights = self.output(hidden)
        return class_weights

    def compose(self,word1, word2, training):
        composed_phrase = transweigh(word1=word1, word2=word2, transformation_tensor=self.transformation_tensor,
                                     transformation_bias=self.transformation_bias, combining_bias=self.combining_bias,
                                     combining_tensor=self.combining_tensor, dropout_rate=self.dropout_rate,
                                     training=training)
        if self.normalize_embeddings:
            composed_phrase = F.normalize(composed_phrase, p=2, dim=1)
        return composed_phrase

    #@property
    #....
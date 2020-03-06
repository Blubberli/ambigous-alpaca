import torch
from torch import nn
import scripts.composition_functions as comp_functions
import torch.nn.functional as F


class PhraseContextClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, forward_hidden_dim, label_nr, dropout_rate):
        """
        This class contains the Phrase And Context Classifier. As input the classifier retrieves a sequence of word
        vectors (= context) and two words of a phrase. The sequence is at first encoded with a bi-directional LSTM.
        The encoded sequence and the phrase (concatenated word vectors) are then fed through a nonlinear classifier.
        :param embedding_dim: embedding dimension of a single word
        :param hidden_size: hidden layer size of the LSTM
        :param num_layers: if more than one, several LSTM layers are stacked upon each other
        :param forward_hidden_dim: the hidden layer size of the forward classifier
        :param label_nr: the number of classes for classification
        :param dropout_rate: the dropout rate that is applied to the forward classifier
        """
        super(PhraseContextClassifier, self).__init__()
        self._hidden_size = hidden_size
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._forward_hidden_dim = forward_hidden_dim
        self._label_nr = label_nr
        self._dropout_rate = dropout_rate
        self._lstm = nn.LSTM(self.embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self._hidden_layer = nn.Linear(2 * embedding_dim + hidden_size * 2, forward_hidden_dim)
        self._output_layer = nn.Linear(forward_hidden_dim, label_nr)

    def forward(self, batch, training):
        # context size = batchsize x max len x embedding dim
        # convert the padded context into a packed sequence such that the padded vectors are not shown to the LSTM
        # context_packed = sum of all seq lenghts, embedding_dim
        # batch_sizes = column-wise (how many real elements do I have?)
        device = batch["device"]
        context_packed = nn.utils.rnn.pack_padded_sequence(batch["seq"].to(device), batch["seq_lengths"].to(device), batch_first=True,
                                                           enforce_sorted=False)
        self.lstm.to(device)
        # forward propagate LSTM, initial states are set to 0 per default
        # out = sum(seq_lenghts), hidden*2
        out, hidden = self.lstm(context_packed)

        # unpack the sequence
        # out: tensor of shape (batch_size, max_seq_length, hidden_size*2)
        out, hidden = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]

        # concat the word vectors into phrase
        word_composed = comp_functions.concat(batch["w1"].to(device), batch["w2"].to(device), axis=1)

        # concate phrase with encoded sequence and send through forward
        context_phrase = torch.cat((word_composed, out), 1)
        x = F.relu(self._hidden_layer(context_phrase))
        x = F.dropout(x, training=training, p=self.dropout_rate)
        return self._output_layer(x)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def embedding_dim(self):
        return self._embedding_dim

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def forward_hidden_dim(self):
        return self._forward_hidden_dim

    @property
    def label_nr(self):
        return self._label_nr

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @property
    def lstm(self):
        return self._lstm


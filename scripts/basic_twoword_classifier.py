import torch.nn as nn
import torch.nn.functional as F
import scripts.composition_functions as comp_functions


class BasicTwoWordClassifier(nn.Module):
    """
    this class includes a basic two-word classifier with one hidden layer and one output layer.
    :param input_dim: the dimension of the input vector where only embedding size (2 times the size of the embedding of
    a single word vector) is needed, batch size is implicit
    :param hidden_dim : the dimension of the hidden layer, batch size is implicit
    :param label_nr : the dimension of the output layer, i.e. the number of labels
    """

    def __init__(self, input_dim, hidden_dim, label_nr):
        super(BasicTwoWordClassifier, self).__init__()
        self._hidden_layer = nn.Linear(input_dim, hidden_dim)
        self._output_layer = nn.Linear(hidden_dim, label_nr)

    def forward(self, word1, word2):
        """
        this function takes two words, concatenates them and applies a non-linear matrix transformation (hidden layer)
        Its output is then fed to an output layer. Then it returns the concatenated and transformed vectors.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the transformed vectors after output layer
        """
        word_composed = comp_functions.concat(word1, word2, axis=1)
        x = F.relu(self.hidden_layer(word_composed))
        return self.output_layer(x)

    @property
    def hidden_layer(self):
        return self._hidden_layer

    @property
    def output_layer(self):
        return self._output_layer

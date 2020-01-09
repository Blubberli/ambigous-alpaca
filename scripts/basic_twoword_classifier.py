import torch.nn as nn
import torch.nn.functional as F
import scripts.composition_functions as comp_functions


class BasicTwoWordClassfier(nn.Module):
    """
    this class includes a basic two-word classifier with one hidden layer and one output layer.
    :param input_dim: the dimension of the input vector where only embedding size (2 times the size of the embedding of
    a single word vector) is needed, batch size is implicit
    :param hidden_dim : the dimension of the hidden layer, batch size is implicit
    :param label_nr : the dimension of the output layer, i.e. the number of labels
    """

    def __init__(self, input_dim, hidden_dim, label_nr):
        super(BasicTwoWordClassfier, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, label_nr)

    def forward(self, word1, word2):
        """
        this function takes two words, concatenates them and applies a non-linear matrix transformation. Then
        it returns the concatenated and transformed vectors.
        :param word1: the first word of size batch_size x embedding size
        :param word2: the first word of size batch_size x embedding size
        :return: the transformed vectors after output layer
        """
        word_composed = comp_functions.concat(word1, word2, axis=1)
        x = F.relu(self.hidden_layer(word_composed))
        return F.relu(self.output_layer(x))







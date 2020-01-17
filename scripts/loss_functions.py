import torch.nn.functional as F
import torch


def multi_class_cross_entropy(output, target):
    """
    combines log_softmax and nll_loss in a single function, is used for multiclass classification
    :param output: the input to the loss function is the output as raw, unnormalized scores for each class,
                    is of size (minibatch, C)
    :param target: 1D tensor of size minibatch with range [0,C−1] for each value,
    :return: loss: mean loss of all instances in batch
    """
    assert output.shape[0] == target.shape[0], "target shape is the number of batches in output"
    assert target.dtype != 'float32', "target type has to be \"int\""  # maybe remove or use regex
    loss = F.cross_entropy(output, target)
    return loss


def binary_class_cross_entropy(output, target):
    """
    applies a sigmoid function plus cross-entropy loss, is used for binary classification
    :param output: the input to the loss function is the output as raw, unnormalized scores for each class
    :param target: tensor of the same shape as input
    :return: loss: mean loss of all instances in batch
    """
    assert output.shape == target.shape, "target shape is same as output shape"
    loss = F.binary_cross_entropy(torch.sigmoid(output), target)
    return loss

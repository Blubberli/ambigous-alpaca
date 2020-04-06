import torch.nn.functional as F
import torch


def multi_class_cross_entropy(output, target):
    """
    combines log_softmax and nll_loss in a single function, is used for multiclass classification
    :param output: the input to the loss function is the output as raw, unnormalized scores for each class,
                    is of size (minibatch, C) and of type float
    :param target: 1D tensor of size minibatch with range [0,C−1] for each value and is of type int
    :return: loss: mean loss of all instances in batch
    """
    assert output.shape[0] == target.shape[0], "target shape is the number of batches in output"

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


def get_loss_cosine_distance(original_phrase, composed_phrase, dim=1, normalize=False):
    """
    Computes the cosine distance between two given phrases. The distance can be used to pretrain a composition model.
    :param original_phrase: the gold standard representations
    :param composed_phrase: the representations retrieved from the composition model
    :param dim: along which dimension to compute the distance and to normalize, default=1 for two batches
    :param normalize: Whether the input embeddings should be normalized, the function expects the input to already be
    normalized
    :return: The averaged cosine distance for one batch
    """
    assert original_phrase.shape == composed_phrase.shape, "shapes of original and composed phrase have to be the same"
    if normalize:
        original_phrase = F.normalize(original_phrase, p=2, dim=dim)
        composed_phrase = F.normalize(original_phrase, p=2, dim=dim)
    cosine_distances = 1 - F.cosine_similarity(original_phrase, composed_phrase, dim)
    total = torch.sum(cosine_distances)
    return total / original_phrase.shape[0]

import torch.nn.functional as F

def multi_class_cross_entropy(output, target):
    """
    This criterion combines log_softmax and nll_loss in a single function.
    :param output: the input to the loss function is the output as raw, unnormalized scores for each class,
                    is of size (minibatch, C)
    :param target: 1D tensor of size minibatch with range [0,C−1] for each value,
    :return: loss: mean loss of all instances in batch
    """
    assert output.shape[0] == target.shape[0], "target shape is the number of batches in output"
    assert target.dtype != 'float32', "target type has to be \"int\""
    loss = F.cross_entropy(output, target)
    return loss


# to be done
def binary_class_cross_entropy(output, target):
    """
    :param output: the input to the loss function is the output as raw, unnormalized scores for each class
    :param target: tensor of the same shape as input
    :return: loss: mean loss of all instances in batch
    """
    assert output.shape[0] == target.shape, "target shape is the number of batches in output" # maybe change
    loss = F.binary_cross_entropy(F.sigmoid(output), target)
    return loss









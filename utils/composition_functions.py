import torch
import torch.functional as Fn
import torch.nn.functional as F


def transweigh_weight(trans_uv, combining_tensor, combining_bias):
    """
    This function takes all transformed composed phrases as input and combines them into one final representation via
    tensor double contraction.
    :param trans_uv: The a tensor containing all the transformed input vectors : torch tensor of [batch_size,
    no_transformations, embedding_dim]
    :param combining_tensor: A third-order tensor that can be used to weigh each element in each transformation
    differently : a torch tensor of [no_transformations, embedding_dim, embedding_dim]
    :param combining_bias: a bias that is added to the final representation : torch tensor of [embedding_dim]
    :return: a final representation that is informed by all elements of all transformed input representations :
    a torch tensor of [batch_size x embedding_dim]
    """
    weighted_uv = Fn.tensordot(trans_uv, combining_tensor, dims=[[1, 2], [0, 1]])
    weighted_transformations = weighted_uv + combining_bias
    return weighted_transformations


def tw_transform(word1, word2, transformation_tensor, transformation_bias):
    """
    This function applies a transformation to the representation of word1 and the representation of word2. The The
    concatenated input features are transformed via a third-order tensor trough double tensor-contraction. A bias is
    added afterwards.
    :param word1: A feature representation of word1: a torch tensor of [batchsize x embedding_dim]
    :param word2: A feature representation of word2: a torch tensor of [batchsize x embedding_dim]
    :param transformation_tensor: A third-order tensor, which contains several tranformation matrices:
    a torch tensor of [no_transformations, 2*embedding_dim, embedding_dim]
    :param transformation_bias: a bias, applied after every transformation:
    a torch tensor of [no_transformations, embedding_dim]
    :return: several composed phrases, each representing a different transformation of the input vectors plus bias:
    a torch tensor of [no_transformations, embedding_dim]
    """
    # concat the two word embedding matrices
    # batch_size x 2embedding_size
    uv = concat(word1, word2, axis=1)
    # create all the transformations of the input vectors u and v
    # batch_size x 2embedding_size * transformations x 2embedding_size x embedding_size ->
    # batch_size x transformations x embedding_size
    trans_uv = Fn.tensordot(uv, transformation_tensor, dims=[[1], [1]])
    # add biases
    # batch_size x transformations x embedding_size + transformations x embedding_size (auto broadcast)
    trans_uv_bias_sum = trans_uv + transformation_bias
    return trans_uv_bias_sum


def transweigh(word1, word2, transformation_tensor, transformation_bias, combining_tensor, combining_bias, training,
               dropout_rate):
    """
    In this function multiple, different affine transformations are applied to the same input vectors. All different
    combinations of the input vectors are composed into one final representation.
    :param word1: A feature representation of word1: a torch tensor of [batchsize x embedding_dim]
    :param word2: A feature representation of word2: a torch tensor of [batchsize x embedding_dim]
    :param transformation_tensor: A third-order tensor, which contains several tranformation matrices: a torch tensor
    of [no_transformations, 2*embedding_dim, embedding_dim]
    :param transformation_bias: a bias, applied after every transformation: a torch tensor of [embedding_dim]
    :param combining_tensor: A third-order tensor that can be used to weigh each element in each transformation
    differently : a torch tensor of [no_transformations, embedding_dim, embedding_dim]
    :param combining_bias: a bias that is added to the final representation : torch tensor of [embedding_dim]
    :param training: indicates whether dropout should be applied because of training mode: boolean
    :param dropout_rate: if dropout should be applied, a floating point number > 0 and < 1.
    :return: the final composed phrase : a torch tensor of [batch_size, embedding_dim]
    """
    # transform the input
    transformed_input = tw_transform(word1, word2, transformation_tensor, transformation_bias)

    # apply dropout and non-linearity
    reg_uv = F.relu(transformed_input)
    reg_uv = F.dropout(reg_uv, training=training, p=dropout_rate)
    # weigh the transformations and combine them into one representation
    weighted_uv = transweigh_weight(reg_uv, combining_tensor, combining_bias)
    return weighted_uv


def concat(word1, word2, axis):
    """
    returns the concatenation of two vectors [word1;word2]. This can be used as a default "non-composition function"
    :param word1: a tensor with any shape
    :param word2: a tensor with the same shape as word1
    :param axis: on which axis the two tensors should be concatenated
    :return: a new tensor that is the concatenation of word1 and word2
    """
    assert word1.shape == word2.shape, "can only concatenate two tensors of the same shape"
    assert axis <= len(word1.shape), "the given axis is out of bounds"
    return torch.cat((word1, word2), axis)


def full_additive(modifier_matrix, modifier_vector, head_matrix, head_vector):
    """
    composes two vectors the following way: modifier_matrix * modifier_vector + head_matrix * head_vector
    :param modifier_matrix: a torch tensor
    :param modifier_vector: a torch tensor
    :param head_matrix: a torch tensor
    :param head_vector: a torch tensor
    :return: a new tensor that is the composed form of modifier_vector and head_vector
    """
    transformed_modifier = modifier_vector.matmul(modifier_matrix.t())
    transformed_head = head_vector.matmul(head_matrix.t())
    return transformed_modifier + transformed_head

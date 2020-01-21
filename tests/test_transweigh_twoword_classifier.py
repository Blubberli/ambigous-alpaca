import numpy as np
import torch
from torch import optim
import unittest
from scripts import TransweighTwoWordClassifier
from scripts import binary_class_cross_entropy


class TransweighTwoWordClassifierTest(unittest.TestCase):

    def setUp(self):
        self.model = TransweighTwoWordClassifier(input_dim=4, hidden_dim=2, label_nr=1, transformations=5,
                                                 dropout_rate=0.1,
                                                 normalize_embeddings=True)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.w1 = torch.from_numpy(np.array([[0.3, 0.1, 0.2, 0.4], [0.0, 0.5, 0.0, 0.5]], dtype=np.float32))
        self.w2 = torch.from_numpy(np.array([[0.5, 0.3, 0.7, 0.1], [0.1, 0.5, 0.6, 0.1]], dtype=np.float32))
        self.label = torch.from_numpy(np.array([[1.0], [0.0]], dtype=np.float32))
        self.model = self.model.float()

    def test_model(self):
        self.optimizer.zero_grad()
        self.model.float()
        output = self.model(self.w1, self.w2, True)
        loss = binary_class_cross_entropy(output, self.label)
        loss.backward()
        self.optimizer.step()
        print(loss.item())

    def test_trainable_parameters(self):
        variables = ["_transformation_tensor", "_transformation_bias", "_combining_tensor", "_combining_bias",
                     "_hidden.weight", "_hidden.bias", "_output.weight", "_output.bias"]
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                np.testing.assert_equal(name in variables, True)


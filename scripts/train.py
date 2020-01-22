from scripts.composition_functions import transweigh, transweigh_weight, tw_transform, concat
from scripts.basic_twoword_classifier import BasicTwoWordClassifier
from scripts.feature_extractor_contextualized import BertExtractor
from scripts.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy
import torch
import numpy as np


def main():
    batch_wordone = torch.from_numpy(
            np.array([[5.0, 5.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]))
    batch_wordtwo = torch.from_numpy(
            np.array([[3.0, 4.0, 2.0], [1.0, 5.0, 2.0], [3.0, 3.0, 3.0]]))

    classifier = BasicTwoWordClassifier(input_dim=3, hidden_dim=3, label_nr=3)
    loss = classifier(batch_wordone, batch_wordtwo)
    loss = loss.backward()
    print(loss)
if __name__ == "__train__":
    main()
from scripts.basic_twoword_classifier import BasicTwoWordClassifier
from scripts.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy
import torch
from torch import optim
import numpy as np
import logging
import logging.config
from scripts.logger_config import create_config

#from pytorchtools import EarlyStopping


# add method that saves predictions to file


def train_binary():
    #logger.info("setting up classifier with shape %s" % batch_wordtwo.shape[1])
    #logger.info("%d training batches" % batch_wordtwo.shape[0])  # ....and so on....



    # classifier = BasicTwoWordClassifier(input_dim=batch_wordtwo.shape[1] * 2, hidden_dim=batch_wordone.shape[0],
    #                                     label_nr=len(target))
    # optimizer = optim.Adam(classifier.parameters(), lr=0.1)
    # for epoch in range(20):
    #     out = classifier(batch_wordone, batch_wordtwo)
    #     loss = multi_class_cross_entropy(out, target)
    #     loss.backward()
    #     optimizer.step()
    #     print(loss)

def train_multiclass():
    pass

def predict():
    pass


def save_predictions():
    pass

def do_eval():
    pass


def main():

    #instead read json file
    batch_wordone = torch.from_numpy(
            np.array([[5.0, 2.0, 3.0], [0.0, 3.0, 1.0], [1.0, 0.0, 6.0]], dtype="float32"))
    batch_wordtwo = torch.from_numpy(
            np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 2.0, 2.0]],dtype="float32"))
    target = torch.from_numpy(
            np.array([0,1,2],dtype="int64"))
    output_path = "/Users/ehuber/Documents/ambiguous_alpaca/ambigous-alpaca/trylog.txt" # will be taken from json file

    logging.config.dictConfig(create_config(output_path))
    logger = logging.getLogger("train")
    #logging.info().. add information of that was json file

if __name__ == "__main__":


    main()

    # read in arguments and save them into a configuration object
    # classifier, hidden_layer dropout, path to dataset, hidden dimension, binary classification, no of epochs, gpu, learningrate
    # initialize model, optimizer, create logger
    # training loop for binary classification (+ validation)
    # training loop for multiclass classification (+ validation)
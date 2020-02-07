from scripts.basic_twoword_classifier import BasicTwoWordClassifier
from scripts.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy
import torch
from torch import optim
import numpy as np
import logging
import logging.config
from scripts.logger_config import create_config
import argparse
import json
from pathlib import Path
import time
from scripts.data_loader import SimplePhraseContextualizedDataset, SimplePhraseStaticDataset
from torch.utils.data import DataLoader

from pytorchtools import EarlyStopping


from scripts.training_utils import init_classifier
# add method that saves predictions to file


def train_binary(config, data):



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

def train_multiclass(config, data):

    model = init_classifier(config)
    optimizer = optim.Adam(model.parameters(), lr=config["trainer"]["lr"]) # or make an if statement for choosing an optimizer
    train_losses = []

    for epoch in range(config["num_epochs"]):
        for word1,word2,labels in data:
            predictions = model(word1, word2)
            loss = multi_class_cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()




def predict():
    pass


def save_predictions():
    pass

def do_eval():
    pass





if __name__ == "__main__":
    # instead read json file
    # batch_wordone = torch.from_numpy(
    #         np.array([[5.0, 2.0, 3.0], [0.0, 3.0, 1.0], [1.0, 0.0, 6.0]], dtype="float32"))
    # batch_wordtwo = torch.from_numpy(
    #         np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 2.0, 2.0]],dtype="float32"))
    # target = torch.from_numpy(
    #         np.array([0,1,2],dtype="int64"))
    # output_path = "/Users/ehuber/Documents/ambiguous_alpaca/ambigous-alpaca/trylog.txt" # will be taken from json file
    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp.parse_args()
    config = json.loads(argp.path_to_config) # read in arguments and save them into a configuration object
    ts = time.gmtime()
    save_name = format("%s_%s" % (config[""], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))
    path_logfile = str(Path(config["save_path"]).joinpath(save_name + "_log.txt"))
    logging.config.dictConfig(create_config(path_logfile))
    logger = logging.getLogger("train")

    logger.info("Training %s model with %s embeddings. Logging to %s" % (config["model"]["type"], config["feature_extractor"], path_logfile))

    for k,v in config.items():
        logger.info("%s : %s", (k,v))

    #read in data...

    if config["feature_extractor"]["contextualized_embeddings"] is True:
        if config["context"] is False:
            if config["feature_extractor"]["static_embeddings"] is True:
                dataset = SimplePhraseContextualizedDataset(config["train_data_path"], config["contextualized"]["bert"])
    else:
        dataset = SimplePhraseStaticDataset(config["train_data_path"], config["static"]["pretrained_model"])


    #load data with torch Data Loader
    train_loader = DataLoader(dataset, batch_size=config["iterator"]["batch_size"], shuffle=True, num_workers=2) # maybe move to train method
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=0)



    logger.info("%d training batches" % config["iterator"]["batch_size"])
    logger.info("the training data contains %d words" % len(train_loader))


    #train
    if config["model"]["classification"] == "multi":
        train_multiclass(config, train_loader, valid_loader)
    elif config["model"]["classification"] == "binary":
        train_binary(config, train_loader, valid_loader)
    else:
        print("classification has to be specified as either multi or binary")






    # classifier, hidden_layer dropout, path to dataset, hidden dimension, binary classification, no of epochs, gpu, learningrate
    # initialize model, optimizer, create logger
    # training loop for binary classification (+ validation)
    # training loop for multiclass classification (+ validation)
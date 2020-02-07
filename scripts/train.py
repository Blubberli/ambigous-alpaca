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


from scripts.training_utils import init_classifier



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

def train_multiclass(config, train_loader, valid_loader):

    model = init_classifier(config)
    optimizer = optim.Adam(model.parameters(), lr=config["trainer"]["lr"]) # or make an if statement for choosing an optimizer
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    while current_patience < config["patience"]
        for epoch in range(config["num_epochs"]):
            model.train()
            for word1,word2,labels in train_loader:
                predictions = model(word1, word2)
                loss = multi_class_cross_entropy(predictions, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(loss.item())

            model.eval()
            for word1, word2, labels in valid_loader:
                predictions = model(word1, word2)
                loss = multi_class_cross_entropy(predictions, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                valid_losses.append(loss.item())

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            if (lowest_loss - valid_loss > tolerance):
                current_patience = 0
                #here also model should be saved in the future
            else:
                current_patience +=1
            #append average loss to list for all epochs
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            #set back lists for next epoch
            train_losses = []
            valid_losses = []


def predict(): # for test set
    #training = False
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
                dataset_train = SimplePhraseContextualizedDataset(config["train_data_path"], config["contextualized"]["bert"])
                dataset_test = SimplePhraseContextualizedDataset(config["test_data_path"], config["contextualized"]["bert"])
    else:
        dataset_train = SimplePhraseStaticDataset(config["train_data_path"], config["static"]["pretrained_model"])
        dataset_test = SimplePhraseStaticDataset(config["test_data_path"], config["static"]["pretrained_model"])


    #load data with torch Data Loader
    train_loader = DataLoader(dataset_train,
                              batch_size=config["iterator"]["batch_size"],
                              shuffle=True,
                              num_workers=0)
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=config["iterator"]["batch_size"],
                                               shuffle=True,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=config["iterator"]["batch_size"],
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
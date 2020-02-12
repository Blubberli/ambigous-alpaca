from scripts.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy
import torch
from torch import optim
import numpy as np
import logging.config
from scripts.logger_config import create_config
import argparse
import json
from pathlib import Path
import time
from scripts.data_loader import SimplePhraseContextualizedDataset, SimplePhraseStaticDataset
from torch.utils.data import DataLoader
from scripts.training_utils import init_classifier



def train_binary(config, train_loader, valid_loader):
    """
    method to train a binary classification model
    :param config: config json file
    :param train_loader: dataloader torch object with training data
    :param valid_loader: dataloader torch object with validation data
    :return: average train losses of each batch, average validation losses of each batch and trained model
    """
    pass

def train_multiclass(config, train_loader, valid_loader):
    """
    method to train a multiclass classification model
    :param config: config json file
    :param train_loader: dataloader torch object with training data
    :param valid_loader: dataloader torch object with validation data
    :return: average train losses of each batch, average validation losses of each batch and trained model
    """
    model = init_classifier(config)
    optimizer = optim.Adam(model.parameters(), lr=config["trainer"]["optimizer"]["lr"]) # or make an if statement for choosing an optimizer
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(1, config["num_epochs"]+1):
        model.train()
        for word1, word2, labels in train_loader:
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
            valid_losses.append(loss.item())

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        if (lowest_loss - valid_loss > tolerance):
            lowest_loss = valid_loss
            current_patience = 0
            # here also model should be saved in the future
        else:
            current_patience += 1
        if current_patience > config["patience"]:
            break
        # append average loss to list for all epochs
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        # set back lists for next epoch
        train_losses = []
        valid_losses = []
        logger.info("current patience: %d , epoch %d , train_loss: %.5f validation loss: %.5f" %
                    (current_patience, epoch, train_loss, valid_loss))
    return (avg_train_losses, avg_valid_losses, model)


def predict(test_loader, model, config): # for test set
    """
    predicts labels on unseen data (test set)
    :param test_loader: dataloader torch object with test data
    :param model: trained model
    :param config: config: config json file
    :return: predictions and list of losses of each batch
    """
    test_loss = []
    predictions = []
    for word1, word2, labels in test_loader:
        out = model(word1, word2)
        _, prediction = torch.max(out, 1)
        predictions.append(prediction.tolist())
        if config["model"]["classification"] == "multi":
            test_loss.append(multi_class_cross_entropy(out, labels))
        else: #binary
            test_loss.append(binary_class_cross_entropy(out, labels))
    predictions =  [item for sublist in predictions for item in sublist] # flatten list

    return (predictions, test_loss)




def do_eval(predictions, labels):
    correct = 0
    for p, l in zip(predictions, labels):
        if p == l:
            correct +=1
    print((correct/len(predictions)), correct)
    return correct, (correct/len(predictions))


def save_predictions():
    pass


if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp = argp.parse_args()
    with open(argp.path_to_config, 'r') as f: # read in arguments and save them into a configuration object
        config = json.load(f)

        ts = time.gmtime()
        save_name = format("%s_%s" % (config["model"]["type"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))   # change names

        log_file = str(Path(config["logging_path"]).joinpath(save_name + "_log.txt"))  # change location

        logging.config.dictConfig(create_config(log_file))
        logger = logging.getLogger("train")

        logger.info("Training %s model with %s embeddings. Logging to %s" % (config["model"]["type"], config["feature_extractor"], log_file))


        #read in data...

        if config["feature_extractor"]["contextualized_embeddings"] is True:
            if config["context"] is False:
                if config["feature_extractor"]["static_embeddings"] is True:
                    dataset_train = SimplePhraseContextualizedDataset(config["train_data_path"], config["contextualized"]["bert"])
                    dataset_valid = SimplePhraseContextualizedDataset(config["validation_data_path"], config["contextualized"]["bert"])
                    dataset_test = SimplePhraseContextualizedDataset(config["test_data_path"], config["contextualized"]["bert"])
        else:
            dataset_train = SimplePhraseStaticDataset(config["train_data_path"], config["feature_extractor"]["static"]["pretrained_model"])
            dataset_test = SimplePhraseStaticDataset(config["test_data_path"], config["feature_extractor"]["static"]["pretrained_model"])
            dataset_valid = SimplePhraseStaticDataset(config["validation_data_path"],
                                                              config["feature_extractor"]["static"]["pretrained_model"])



        #load data with torch Data Loader
        train_loader = DataLoader(dataset_train,
                                  batch_size=config["iterator"]["batch_size"],
                                  shuffle=True,
                                  num_workers=0)
        # load validation data in batches
        valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                                   batch_size=config["iterator"]["batch_size"],
                                                   shuffle=True,
                                                   num_workers=0)

        # load test data in batches
        test_loader = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=config["iterator"]["batch_size"],
                                                  num_workers=0)


        test_labels = [l for w1,w2,l in dataset_test]


    logger.info("%d training batches" % config["iterator"]["batch_size"])
    logger.info("the training data contains %d words" % len(dataset_train))
    logger.info("the validation data contains %d words" % len(dataset_valid))
    logger.info("the test data contains %d words" % len(dataset_test))


    #train & test and & evaluate
    if config["model"]["classification"] == "multi":
        avg_train_losses, avg_valid_losses, model = train_multiclass(config, train_loader, valid_loader)
        predictions, test_loss = predict(test_loader, model, config)
        accuracy = do_eval(predictions, test_labels)
    elif config["model"]["classification"] == "binary":
        avg_train_losses, avg_valid_losses, model = train_binary(config, train_loader, valid_loader)
        predictions, test_loss = predict(test_loader, model, config)
        accuracy = do_eval(predictions, test_labels)
    else:
        print("classification has to be specified as either multi or binary")


    #comments Neele
    # classifier, hidden_layer dropout, path to dataset, hidden dimension, binary classification, no of epochs, gpu, learningrate
    # initialize model, optimizer, create logger
    # training loop for binary classification (+ validation)
    # training loop for multiclass classification (+ validation)
import argparse
import json
import time
from pathlib import Path
import torch
from torch import optim
import numpy as np
import logging.config
from scripts.logger_config import create_config
from torch.utils.data import DataLoader
from scripts.training_utils import init_classifier, get_datasets, convert_logits_to_binary_predictions
from scripts.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy


def train_binary(config, train_loader, valid_loader):
    """
    method to train a binary classification model
    :param config: config json file
    :param train_loader: dataloader torch object with training data
    :param valid_loader: dataloader torch object with validation data
    :return: the trained model
    """
    model = init_classifier(config)
    optimizer = optim.Adam(model.parameters(),
                           lr=config["trainer"]["optimizer"]["lr"])  # or make an if statement for choosing an optimizer
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    best_epoch = 1
    epoch = 1
    train_loss = 0.0
    best_accuracy = 0.0

    for epoch in range(1, config["num_epochs"] + 1):
        # training loop over all batches
        model.train()
        # these store the losses and accuracies for each batch for one epoch
        train_losses = []
        valid_losses = []
        valid_accuracies = []

        for word1, word2, labels in train_loader:
            out = model(word1, word2, True).squeeze()

            #print(out)
            loss = binary_class_cross_entropy(out, labels.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
        # validation loop over validation batches
        model.eval()
        for word1, word2, labels in valid_loader:
            out = model(word1, word2, False).squeeze()
            predictions = convert_logits_to_binary_predictions(out)
            loss = binary_class_cross_entropy(out, labels.float())
            valid_losses.append(loss.item())
            _, accur = get_accuracy(predictions, labels)
            valid_accuracies.append(accur)

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        valid_accuracy = np.average(valid_accuracies)

        if lowest_loss - valid_loss > tolerance:
            lowest_loss = valid_loss
            best_epoch = epoch
            best_accuracy = valid_accuracy
            current_patience = 0
            # here also model should be saved in the future
        else:
            current_patience += 1
        if current_patience > config["patience"]:
            break

        logger.info("current patience: %d , epoch %d , train loss: %.5f, validation loss: %.5f, accuracy: %.5f " %
                    (current_patience, epoch, train_loss, valid_loss, valid_accuracy))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f, "
        "best validation accuracy: %.5f " %
        (epoch, train_loss, best_epoch, lowest_loss, best_accuracy))
    return model


def train_multiclass(config, train_loader, valid_loader):
    """
    method to train a multiclass classification model
    :param config: config json file
    :param train_loader: dataloader torch object with training data
    :param valid_loader: dataloader torch object with validation data
    :return: the trained model
    """
    model = init_classifier(config)
    optimizer = optim.Adam(model.parameters(),
                           lr=config["trainer"]["optimizer"]["lr"])  # or make an if statement for choosing an optimizer
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    best_epoch = 1
    epoch = 1
    train_loss = 0.0
    best_accuracy = 0.0

    for epoch in range(1, config["num_epochs"] + 1):
        # training loop over all batches
        model.train()
        # these store the losses and accuracies for each batch for one epoch
        train_losses = []
        valid_losses = []
        valid_accuracies = []
        for word1, word2, labels in train_loader:
            out = model(word1, word2, True)
            loss = multi_class_cross_entropy(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        model.eval()
        for word1, word2, labels in valid_loader:
            out = model(word1, word2, False)
            _, predictions = torch.max(out, 1)
            loss = multi_class_cross_entropy(out, labels)
            valid_losses.append(loss.item())
            _, accur = get_accuracy(predictions.tolist(), labels)
            valid_accuracies.append(accur)

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        valid_accuracy = np.average(valid_accuracies)

        if lowest_loss - valid_loss > tolerance:
            lowest_loss = valid_loss
            best_epoch = epoch
            best_accuracy = valid_accuracy
            current_patience = 0
            # here also model should be saved in the future
        else:
            current_patience += 1
        if current_patience > config["patience"]:
            break

        logger.info("current patience: %d , epoch %d , train loss: %.5f, validation loss: %.5f, accuracy: %.5f " %
                    (current_patience, epoch, train_loss, valid_loss, valid_accuracy))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f, "
        "best validation accuracy: %.5f " %
        (epoch, train_loss, best_epoch, lowest_loss, best_accuracy))
    return model


def predict(test_loader, model, config):  # for test set
    """
    predicts labels on unseen data (test set)
    :param test_loader: dataloader torch object with test data
    :param model: trained model
    :param config: config: config json file
    :return: predictions for the given dataset, the loss and accuracy over the whole dataset
    """
    test_loss = []
    predictions = []
    accuracy = []
    for word1, word2, labels in test_loader:
        out = model(word1, word2, False)
        if config["model"]["classification"] == "multi":
            test_loss.append(multi_class_cross_entropy(out, labels).item())
            _, prediction = torch.max(out, 1)
            prediction = prediction.tolist()
        else:  # binary
            test_loss.append(binary_class_cross_entropy(out.squeeze(), labels.float()).item())
            prediction = convert_logits_to_binary_predictions(out)
        _, accur = get_accuracy(prediction, labels)
        predictions.append(prediction)
        accuracy.append(accur)
    predictions = [item for sublist in predictions for item in sublist]  # flatten list
    return predictions, np.average(test_loss), np.average(accuracy)


def get_accuracy(predictions, labels):
    correct = 0
    for p, l in zip(predictions, labels):
        if p == l:
            correct += 1
    accuracy = correct / len(predictions)
    return correct, accuracy


def save_predictions():
    pass


if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp = argp.parse_args()

    with open(argp.path_to_config, 'r') as f:  # read in arguments and save them into a configuration object
        config = json.load(f)

    ts = time.gmtime()

    # if name is specified choose specified name to save logging file, else use default name
    if config["save_name"] == "":
        save_name = format(
            "%s_%s" % (config["model"]["type"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names
    else:
        save_name = format("%s_%s" % (config["save_name"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names

    log_file = str(Path(config["logging_path"]).joinpath(save_name + "_log.txt"))  # change location

    logging.config.dictConfig(create_config(log_file))
    logger = logging.getLogger("train")
    logger.info("Training %s model with %s embeddings. \n Logging to %s" % (
        config["model"]["type"], config["feature_extractor"], log_file))

    # set random seed
    np.random.seed(config["seed"])
    # read in data...
    dataset_train, dataset_valid, dataset_test = get_datasets(config)

    # load data with torch Data Loader
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

    test_labels = [l for w1, w2, l in dataset_test]
    model = None

    logger.info("%d training batches" % config["iterator"]["batch_size"])
    logger.info("the training data contains %d words" % len(dataset_train))
    logger.info("the validation data contains %d words" % len(dataset_valid))
    logger.info("the test data contains %d words" % len(dataset_test))

    # train
    if config["model"]["classification"] == "multi":
        model = train_multiclass(config, train_loader, valid_loader)
    elif config["model"]["classification"] == "binary":
        model = train_binary(config, train_loader, valid_loader)
    else:
        print("classification has to be specified as either multi or binary")

    # test and & evaluate
    # loading best model from checkpoint model = load_model()
    if model:
        logger.info("generating predictions for validation data...")
        valid_predictions, valid_loss, valid_acc = predict(valid_loader, model, config)
        logger.info("validation loss: %.5f, validation accuracy : %.5f" %
                    (valid_loss, valid_acc))
        if config["eval_on_test"]:
            logger.info("generating predictions for test data...")
            test_predictions, test_loss, test_acc = predict(test_loader, model, config)
            logger.info("test loss: %.5f, test accuracy : %.5f" %
                        (test_loss, test_acc))
    else:
        logging.error("model could not been loaded correctly")

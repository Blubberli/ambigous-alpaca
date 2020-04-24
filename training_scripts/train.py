import argparse
import json
import time
from pathlib import Path
import torch
from torch import optim
import numpy as np
from sklearn.metrics import f1_score
import logging.config
from utils.logger_config import create_config
from torch.utils.data import DataLoader
from utils.plot_utils import plot_learning_curves
from utils.training_utils import init_classifier, get_datasets, convert_logits_to_binary_predictions
from utils.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy


def train_binary(config, train_loader, valid_loader, model_path, device):
    """
    method to train a binary classification model
    :param config: config json file
    :param train_loader: dataloader torch object with training data
    :param valid_loader: dataloader torch object with validation data
    :return: the trained model
    """
    model = init_classifier(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters())  # or make an if statement for choosing an optimizer
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    best_epoch = 1
    epoch = 1
    train_loss = 0.0
    best_accuracy = 0.0
    best_f1 = 0.0
    total_train_losses = []
    total_val_losses = []
    early_stopping_criterion = config["validation_metric"]

    for epoch in range(1, config["num_epochs"] + 1):
        # training loop over all batches
        model.train()
        # these store the losses and accuracies for each batch for one epoch
        train_losses = []
        valid_losses = []
        valid_accuracies = []
        valid_f1_scores = []

        # for word1, word2, labels in train_loader:
        for batch in train_loader:
            batch["device"] = device
            out = model(batch).squeeze().to("cpu")
            loss = binary_class_cross_entropy(out, batch["l"].float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
        # validation loop over validation batches
        model.eval()
        for batch in valid_loader:
            batch["device"] = device
            out = model(batch).squeeze().to("cpu")
            predictions = convert_logits_to_binary_predictions(out)
            loss = binary_class_cross_entropy(out, batch["l"].float())
            valid_losses.append(loss.item())
            _, accur = get_accuracy(predictions, batch["l"])
            f1 = f1_score(y_true=batch["l"], y_pred=predictions)
            valid_accuracies.append(accur)
            valid_f1_scores.append(f1)

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        valid_accuracy = np.average(valid_accuracies)
        valid_f1 = np.average(valid_f1_scores)
        total_val_losses.append(valid_loss)
        total_train_losses.append(train_loss)
        # stop when f1 score is the highest
        if early_stopping_criterion == "f1":
            if valid_f1 > best_f1 - tolerance:
                lowest_loss = valid_loss
                best_f1 = valid_f1
                best_epoch = epoch
                best_accuracy = valid_accuracy
                current_patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                current_patience += 1
        # stop when loss is the lowest
        else:
            if lowest_loss - valid_loss > tolerance:
                lowest_loss = valid_loss
                best_epoch = epoch
                best_accuracy = valid_accuracy
                best_f1 = valid_f1
                current_patience = 0
                torch.save(model.state_dict(), model_path)

            else:
                current_patience += 1
        if current_patience > config["patience"]:
            break

        logger.info(
            "current patience: %d , epoch %d , train loss: %.5f, validation loss: %.5f, accuracy: %.5f, f1 score: %5f" %
            (current_patience, epoch, train_loss, valid_loss, valid_accuracy, valid_f1))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f, "
        "best f1 score: %5f"
        "best validation accuracy: %.5f " %
        (epoch, train_loss, best_epoch, lowest_loss, best_accuracy, best_f1))
    if config["plot_curves"]:
        path = str(Path(config["model_path"]).joinpath(save_name + "_learning_curves.png"))
        plot_learning_curves(training_losses=total_train_losses, validation_losses=total_val_losses, save_path=path)


def train_multiclass(config, train_loader, valid_loader, model_path, device):
    """
    method to train a multiclass classification model
    :param config: config json file
    :param train_loader: dataloader torch object with training data
    :param valid_loader: dataloader torch object with validation data
    :return: the trained model
    """
    model = init_classifier(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters())  # or make an if statement for choosing an optimizer
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    best_epoch = 1
    epoch = 1
    train_loss = 0.0
    best_accuracy = 0.0
    best_f1 = 0.0
    early_stopping_criterion = config["validation_metric"]
    total_train_losses = []
    total_val_losses = []
    for epoch in range(1, config["num_epochs"] + 1):
        # training loop over all batches
        model.train()
        # these store the losses and accuracies for each batch for one epoch
        train_losses = []
        valid_losses = []
        valid_accuracies = []
        valid_f1_scores = []
        for batch in train_loader:
            batch["device"] = device
            out = model(batch).to("cpu")
            loss = multi_class_cross_entropy(out, batch["l"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        model.eval()
        for batch in valid_loader:
            batch["device"] = device
            out = model(batch).to("cpu")
            _, predictions = torch.max(out, 1)
            loss = multi_class_cross_entropy(out, batch["l"])
            valid_losses.append(loss.item())
            _, accur = get_accuracy(predictions.tolist(), batch["l"])
            f1 = f1_score(y_true=batch["l"], y_pred=predictions, average="weighted")
            valid_accuracies.append(accur)

            valid_f1_scores.append(f1)

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        valid_accuracy = np.average(valid_accuracies)
        valid_f1 = np.average(valid_f1_scores)
        total_train_losses.append(train_loss)
        total_val_losses.append(valid_loss)

        # stop when f1 score is the highest
        if early_stopping_criterion == "f1":
            if valid_f1 > best_f1 - tolerance:
                lowest_loss = valid_loss
                best_f1 = valid_f1
                best_epoch = epoch
                best_accuracy = valid_accuracy
                current_patience = 0
                torch.save(model.state_dict(), model_path)
            else:
                current_patience += 1
        # stop when loss is the lowest
        else:
            if lowest_loss - valid_loss > tolerance:
                lowest_loss = valid_loss
                best_epoch = epoch
                best_accuracy = valid_accuracy
                best_f1 = valid_f1
                current_patience = 0
                torch.save(model.state_dict(), model_path)

            else:
                current_patience += 1
        if current_patience > config["patience"]:
            break

        logger.info(
            "current patience: %d , epoch %d , train loss: %.5f, validation loss: %.5f, accuracy: %.5f, f1 score: %5f" %
            (current_patience, epoch, train_loss, valid_loss, valid_accuracy, valid_f1))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f, "
        "best validation accuracy: %.5f, best f1 score %.5f" %

        (epoch, train_loss, best_epoch, lowest_loss, best_accuracy, best_f1))
    if config["plot_curves"]:
        path = str(Path(config["model_path"]).joinpath(save_name + "_learning_curves.png"))
        plot_learning_curves(training_losses=total_train_losses, validation_losses=total_val_losses, save_path=path)


def predict(test_loader, model, config, device):  # for test set
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
    f1_scores = []
    model.to(device)
    for batch in test_loader:
        batch["device"] = device
        out = model(batch).to("cpu")
        if config["model"]["classification"] == "multi":
            test_loss.append(multi_class_cross_entropy(out, batch["l"]).item())
            _, prediction = torch.max(out, 1)
            prediction = prediction.tolist()
        else:  # binary
            test_loss.append(binary_class_cross_entropy(out.squeeze(), batch["l"].float()).item())
            prediction = convert_logits_to_binary_predictions(out)
        _, accur = get_accuracy(prediction, batch["l"])
        f1 = f1_score(y_pred=prediction, y_true=batch["l"], average="weighted")
        predictions.append(prediction)
        accuracy.append(accur)
        f1_scores.append(f1)
    predictions = [item for sublist in predictions for item in sublist]  # flatten list
    return predictions, np.average(test_loss), np.average(accuracy), np.average(f1_scores)


def get_accuracy(predictions, labels):
    """
    Given an list of class predictions and a list of labels, this method returns the accuracy
    :param predictions: a list of class predictions [int]
    :param labels: a list of labels [int]
    :return: [float] accuracy for given predictions and true class labels
    """
    correct = 0
    for p, l in zip(predictions, labels):
        if p == l:
            correct += 1
    accuracy = correct / len(predictions)
    return correct, accuracy


def save_predictions(predictions, path):
    np.save(file=path, arr=np.array(presdictions), allow_pickle=True)


if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp = argp.parse_args()

    with open(argp.path_to_config, 'r') as f:  # read in arguments and save them into a configuration object
        config = json.load(f)

    ts = time.gmtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if name is specified choose specified name to save logging file, else use default name
    if config["save_name"] == "":
        save_name = format(
            "%s_%s" % (config["model"]["type"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names
    else:
        save_name = format("%s_%s" % (config["save_name"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))  # change names

    log_file = str(Path(config["logging_path"]).joinpath(save_name + "_log.txt"))  # change location
    model_path = str(Path(config["model_path"]).joinpath(save_name))
    prediction_path_dev = str(Path(config["model_path"]).joinpath(save_name + "_dev_predictions.npy"))
    prediction_path_test = str(Path(config["model_path"]).joinpath(save_name + "_test_predictions.npy"))

    logging.config.dictConfig(create_config(log_file))
    logger = logging.getLogger("train")
    logger.info("parameter")
    logger.info(str(config))
    logger.info("Training %s model with %s embeddings. \n Logging to %s \n Save model to %s" % (
        config["model"]["type"], config["feature_extractor"], log_file, model_path))

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
                                               shuffle=False,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=config["iterator"]["batch_size"],
                                              shuffle=False,
                                              num_workers=0)

    model = None

    logger.info("%d training batches" % config["iterator"]["batch_size"])
    logger.info("the training data contains %d words" % len(dataset_train))
    logger.info("the validation data contains %d words" % len(dataset_valid))
    logger.info("the test data contains %d words" % len(dataset_test))

    # train
    if config["model"]["classification"] == "multi":
        train_multiclass(config, train_loader, valid_loader, model_path, device)
    elif config["model"]["classification"] == "binary":
        train_binary(config, train_loader, valid_loader, model_path, device)
    else:
        print("classification has to be specified as either multi or binary")

    # test and & evaluate
    logger.info("Loading best model from %s", model_path)
    valid_model = init_classifier(config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()
    if valid_model:
        logger.info("generating predictions for validation data...")
        valid_predictions, validation_loss, valid_acc, valid_f1 = predict(valid_loader, valid_model, config, device)
        save_predictions(valid_predictions, prediction_path_dev)
        logger.info("validation loss: %.5f, validation accuracy : %.5f, valid f1 : %.5f" %
                    (validation_loss, valid_acc, valid_f1))
        if config["eval_on_test"]:
            logger.info("generating predictions for test data...")
            test_predictions, test_loss, test_acc, test_f1 = predict(test_loader, valid_model, config, device)
            save_predictions(test_predictions, prediction_path_test)
            logger.info("test loss: %.5f, test accuracy : %.5f, test f1 : %.5f" %
                        (test_loss, test_acc, test_f1))
    else:
        logging.error("model could not been loaded correctly")

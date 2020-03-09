import argparse
import time
import json
from pathlib import Path
import logging.config
from scripts.logger_config import create_config
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from scripts.training_utils import init_classifier, get_datasets
from scripts.loss_functions import get_loss_cosine_distance


def pretrain(config, train_loader, valid_loader, model_path, device):
    """
        method to pretrain a composition model
        :param config: config json file
        :param train_loader: dataloader torch object with training data
        :param valid_loader: dataloader torch object with validation data
        :return: the trained model
        """
    model = init_classifier(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config["trainer"]["optimizer"]["lr"])
    current_patience = 0
    tolerance = 1e-5
    lowest_loss = float("inf")
    best_epoch = 1
    epoch = 1
    train_loss = 0.0
    for epoch in range(1, config["num_epochs"] + 1):
        # training loop over all batches
        model.train()
        # these store the losses and accuracies for each batch for one epoch
        train_losses = []
        valid_losses = []
        # for word1, word2, labels in train_loader:
        for batch in train_loader:
            batch["device"] = device
            out = model(batch).squeeze().to("cpu")
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
            # validation loop over validation batches
            model.eval()
        for batch in valid_loader:
            batch["device"] = device
            out = model(batch).squeeze().to("cpu")
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
            valid_losses.append(loss.item())

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        if lowest_loss - valid_loss > tolerance:
            lowest_loss = valid_loss
            best_epoch = epoch
            current_patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            current_patience += 1
        if current_patience > config["patience"]:
            break

        logger.info("current patience: %d , epoch %d , train loss: %.5f, validation loss: %.5f" %
                    (current_patience, epoch, train_loss, valid_loss))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f" %
        (epoch, train_loss, best_epoch, lowest_loss))


def predict(test_loader, model, device):
    """
    predicts labels on unseen data (test set)
    :param test_loader: dataloader torch object with test data
    :param model: trained model
    :param config: config: config json file
    :return: predictions for the given dataset, the loss and accuracy over the whole dataset
    """
    test_loss = []
    predictions = []
    orig_phrases = []
    model.to(device)
    for batch in test_loader:
        batch["device"] = device
        out = model(batch).squeeze().to("cpu")
        loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
        predictions.append(out)
        test_loss.append(loss.item())
        orig_phrases.append(batch["phrase"])
    predictions = [item for sublist in predictions for item in sublist]  # flatten list
    orig_phrases = [item for sublist in orig_phrases for item in sublist]
    return predictions, np.average(test_loss), orig_phrases


def save_predictions(predictions, orig_phrases, path):
    word2index = dict(zip(orig_phrases, range(0, len(orig_phrases))))
    vocab = ffp.vocab.SimpleVocab(words=orig_phrases, index=word2index)
    storage = ffp.storage.NdArray(np.array(predictions))
    fifu_embeddings = ffp.embeddings.Embeddings(vocab=vocab, storage=storage)
    fifu_embeddings.write(path)


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
    prediction_path_dev = str(Path(config["model_path"]).joinpath(save_name + "_dev_predictions.fifu"))
    prediction_path_test = str(Path(config["model_path"]).joinpath(save_name + "_test_predictions.fifu"))

    logging.config.dictConfig(create_config(log_file))
    logger = logging.getLogger("train")
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
                                               shuffle=True,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=config["iterator"]["batch_size"],
                                              num_workers=0)

    model = None

    logger.info("%d training batches" % config["iterator"]["batch_size"])
    logger.info("the training data contains %d words" % len(dataset_train))
    logger.info("the validation data contains %d words" % len(dataset_valid))
    logger.info("the test data contains %d words" % len(dataset_test))

    # train
    pretrain(config, train_loader, valid_loader, model_path, device)

    # test and & evaluate
    logger.info("Loading best model from %s", model_path)
    valid_model = init_classifier(config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()
    if valid_model:
        logger.info("generating predictions for validation data...")
        valid_predictions, valid_loss, valid_phrases = predict(valid_loader, valid_model, config)
        save_predictions(predictions=valid_predictions, orig_phrases=valid_phrases, path=prediction_path_dev)
        logger.info("validation loss: %.5f" % (valid_loss))
        if config["eval_on_test"]:
            logger.info("generating predictions for test data...")
            test_predictions, test_loss, test_phrases = predict(test_loader, valid_model, config)
            save_predictions(predictions=test_predictions, orig_phrases=test_phrases, path=prediction_path_test)
            logger.info("test loss: %.5f" % (test_loss))
    else:
        logging.error("model could not been loaded correctly")
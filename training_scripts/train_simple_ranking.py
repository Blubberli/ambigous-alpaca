import argparse
import time
import json
from pathlib import Path
import logging.config
from utils.logger_config import create_config
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.training_utils import init_classifier, get_datasets
from utils.data_loader import extract_all_labels, extract_all_words
from utils.loss_functions import get_loss_cosine_distance
import ffp
from utils import BertExtractor, StaticEmbeddingExtractor
from training_scripts.nearest_neighbour import NearestNeigbourRanker


def train(config, train_loader, valid_loader, model_path, device):
    """
        method to pretrain a composition model
        :param config: config json file
        :param train_loader: dataloader torch object with training data
        :param valid_loader: dataloader torch object with validation data
        :return: the trained model
        """
    model = init_classifier(config)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
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
        for pred in out:
            predictions.append(pred.detach().numpy())
        loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
        test_loss.append(loss.item())
        orig_phrases.append(batch["phrase"])
    orig_phrases = [item for sublist in orig_phrases for item in sublist]
    predictions = np.array(predictions)
    return predictions, np.average(test_loss), orig_phrases


def save_predictions(predictions, path):
    np.save(file=path, arr=predictions, allow_pickle=True)


def save_predictions_labels(predictions, labels, path):
    label_vocab = [labels[i] + str(i) for i in range(len(labels))]
    word2index = dict(zip(label_vocab, range(0, len(label_vocab))))
    vocab = ffp.vocab.SimpleVocab(words=label_vocab, index=word2index)
    storage = ffp.storage.NdArray(predictions)
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
    prediction_path_dev = str(Path(config["model_path"]).joinpath(save_name + "_dev_predictions.npy"))
    prediction_path_test = str(Path(config["model_path"]).joinpath(save_name + "_test_predictions.npy"))
    rank_path_dev = str(Path(config["model_path"]).joinpath(save_name + "_dev_ranks.txt"))
    rank_path_test = str(Path(config["model_path"]).joinpath(save_name + "_test_ranks.txt"))

    modus = config["model"]["classification"]
    assert modus == "pretrain_label" or modus == "pretrain_phrase", "no valid modus specified in config"

    logging.config.dictConfig(create_config(log_file))
    logger = logging.getLogger("train")
    logger.info("Training %s model with %s modus. \n Logging to %s \n Save model to %s" % (
        config["model"]["type"], modus, log_file, model_path))

    # set random seed
    np.random.seed(config["seed"])
    if config["feature_extractor"]["contextualized_embeddings"]:
        bert_parameter = config["feature_extractor"]["contextualized"]["bert"]
        bert_model = bert_parameter["model"]
        max_len = bert_parameter["max_sent_len"]
        lower_case = bert_parameter["lower_case"]
        batch_size = bert_parameter["batch_size"]
        feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                          batch_size=batch_size)
    else:
        feature_extractor = StaticEmbeddingExtractor(
            path_to_embeddings=config["feature_extractor"]["static"]["pretrained_model"])

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
    logger.info("training with the following parameter")
    logger.info(config)
    y_label = None
    if modus == "pretrain_label":
        labels = extract_all_labels(training_data=config["train_data_path"],
                                    validation_data=config["validation_data_path"],
                                    test_data=config["test_data_path"],
                                    separator=config["data_loader"]["separator"]
                                    , label=config["data_loader"]["phrase"])
    else:
        labels = extract_all_words(training_data=config["train_data_path"],
                                   validation_data=config["validation_data_path"],
                                   test_data=config["test_data_path"],
                                   separator=config["data_loader"]["separator"],
                                   modifier=config["data_loader"]["modifier"],
                                   head=config["data_loader"]["head"],
                                   phrase=config["data_loader"]["phrase"])

    # train
    train(config, train_loader, valid_loader, model_path, device)
    # test and & evaluate
    logger.info("Loading best model from %s", model_path)
    valid_model = init_classifier(config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()

    if valid_model:
        logger.info("generating predictions for validation data...")
        valid_predictions, valid_loss, valid_phrases = predict(valid_loader, valid_model, device)
        save_predictions(predictions=valid_predictions, path=prediction_path_dev)
        logger.info("saved predictions to %s" % prediction_path_dev)
        logger.info("validation loss: %.5f" % (valid_loss))
        rank_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=len(dataset_valid), num_workers=0)
        ranker_attribute = NearestNeigbourRanker(path_to_predictions=prediction_path_dev,
                                                 embedding_extractor=feature_extractor,
                                                 data_loader=rank_loader,
                                                 all_labels=labels,
                                                 y_label="phrase", max_rank=1000)
        ranker_attribute.save_ranks(rank_path_dev)

        logger.info("result for learned attribute representation")
        logger.info(ranker_attribute.result)
        logger.info("quartiles : %s" % str(ranker_attribute.quartiles))
        logger.info(
            "precision at rank 1: %.2f; precision at rank 5 %.2f" % (ranker_attribute._map_1, ranker_attribute._map_5))
        logger.info("accuracy: %.2f; f1 score: %.2f" % (ranker_attribute.accuracy, ranker_attribute.f1))

        logger.info("saved ranks to %s" % rank_path_dev)
        if config["eval_on_test"]:
            logger.info("generating predictions for test data...")
            test_predictions, test_loss, test_phrases = predict(test_loader, valid_model, device)
            save_predictions(predictions=test_predictions, path=prediction_path_test)
            logger.info("saved predictions to %s" % prediction_path_test)
            logger.info("test loss: %.5f" % (test_loss))
            rank_loader = torch.utils.data.DataLoader(dataset_test, batch_size=len(dataset_test), num_workers=0)
            ranker_attribute = NearestNeigbourRanker(path_to_predictions=prediction_path_test,
                                                     embedding_extractor=feature_extractor,
                                                     data_loader=rank_loader,
                                                     all_labels=labels,
                                                     y_label="phrase", max_rank=1000)
            ranker_attribute.save_ranks(rank_path_dev)

            logger.info("result for learned attribute representation")
            logger.info(ranker_attribute.result)
            logger.info("quartiles : %s" % str(ranker_attribute.quartiles))
            logger.info(
                "precision at rank 1: %.2f; precision at rank 5 %.2f" % (
                    ranker_attribute._map_1, ranker_attribute._map_5))
            logger.info("accuracy: %.2f; f1 score: %.2f" % (ranker_attribute.accuracy, ranker_attribute.f1))

            logger.info("saved ranks to %s" % rank_path_test)
    else:
        logging.error("model could not been loaded correctly")

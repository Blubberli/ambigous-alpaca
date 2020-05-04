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
from training_scripts.nearest_neighbour import NearestNeigbourRanker
from utils.data_loader import MultiRankingDataset, PretrainCompmodelDataset
from tqdm import trange


def pretrain(pretrain_loader, model, optimizer):
    """
    This function trains the model for one epoch on a given training set
    :param pretrain_loader: a dataloader that contains a specific training set
    :param model: the classifier
    :param optimizer: the optimizer
    :return: trained classifier and optimizer
    """
    pbar = trange(100, desc='Pretrain for one epoch...', leave=True)
    for batch in pretrain_loader:
        batch["device"] = device
        composed, rep1, rep2 = model(batch)
        phrase_loss = get_loss_cosine_distance(composed_phrase=rep1, original_phrase=batch["l"])
        phrase_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.update(100 / len(pretrain_loader))
    return model, optimizer


def train(config, pretrain_1, pretrain_2, train_loader, valid_loader_1, valid_loader_2, model_path, device):
    """
    This method trains a composition model jointly on two tasks. On top of that, the model is pretrained on the more
    general / harder task.
    :param config: the main configuration with the settings that should be used for training
    :param pretrain_loader: a dataloader with the dataset that should be used for pretraining
    :param train_loader: a trainloader that contains a "MultiRankingDataset". This returns two batches, each batch
    contains the instances of one of the two datasets
    :param valid_loader_1: the validation dataset loader of the first task (phrase reconstruction)
    :param valid_loader_2: the validation dataset loader of the second task (attribute composition)
    :param model_path: the path to save the model to
    :param device: the device type (CPU or GPU)
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
    if pretrain_1:
        # use one training set for pretraining
        pretrain_loader = DataLoader(dataset_train_1, batch_size=config_1["iterator"]["batch_size"],
                                     shuffle=True,
                                     num_workers=0)
        model, optimizer = pretrain(pretrain_loader, model, optimizer)
    if pretrain_2:
        pretrain_loader = DataLoader(dataset_train_2, batch_size=config_1["iterator"]["batch_size"],
                                     shuffle=True,
                                     num_workers=0)
        model, optimizer = pretrain(pretrain_loader, model, optimizer)
    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        train_losses = []
        valid_losses_attribute = []
        valid_losses_phrase = []
        for batch_task_1, batch_task_2 in train_loader:
            batch_task_1["device"] = device
            batch_task_2["device"] = device

            composed, rep1, rep2 = model(batch_task_1)
            rep1 = rep1.squeeze().to("cpu")
            phrase_loss = get_loss_cosine_distance(composed_phrase=rep1, original_phrase=batch_task_1["l"])

            composed, rep1, rep2 = model(batch_task_2)
            rep2 = rep2.squeeze().to("cpu")
            attribute_loss = get_loss_cosine_distance(composed_phrase=rep2, original_phrase=batch_task_2["l"])

            loss = attribute_loss + phrase_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())
        for batch in valid_loader_1:
            model.eval()
            batch["device"] = device
            _, out, _ = model(batch)
            out = out.squeeze().to("cpu")
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
            valid_losses_phrase.append(loss.item())
        for batch in valid_loader_2:
            batch["device"] = device
            _, _, out = model(batch)
            out = out.squeeze().to("cpu")
            loss = get_loss_cosine_distance(composed_phrase=out, original_phrase=batch["l"])
            valid_losses_attribute.append(loss.item())

        # calculate average loss and accuracy over an epoch
        train_loss = np.average(train_losses)
        valid_loss_label = np.average(valid_losses_attribute)
        valid_loss_phrase = np.average(valid_losses_phrase)
        total_valid_loss = (valid_loss_label + valid_loss_phrase) / 2

        if lowest_loss - total_valid_loss > tolerance:
            lowest_loss = total_valid_loss
            best_epoch = epoch
            current_patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            current_patience += 1
        if current_patience > config["patience"]:
            break

        logger.info(
            "current patience: %d , epoch %d , train loss: %.3f, validation loss task 1: %.3f, validation loss task "
            "2: %.3f" %
            (current_patience, epoch, train_loss, valid_loss_label, valid_loss_phrase))
    logger.info(
        "training finnished after %d epochs, train loss: %.5f, best epoch : %d , best validation loss: %.5f" %
        (epoch, train_loss, best_epoch, lowest_loss))


def predict(test_loader, model, device):
    """
    predicts labels on unseen data (test set)
    :param test_loader: dataloader torch object with test- or validation data
    :param model: trained model
    :param device: the device
    :return: predictions and losses for the learned attribute and the final composed representation, the original
    phrases
    """
    test_loss_att = []
    test_loss_final = []
    predictions_final_rep = []
    predictions_attribute_rep = []
    orig_phrases = []
    model.to(device)
    for batch in test_loader:
        batch["device"] = device
        composed, rep1, rep2 = model(batch)
        composed = composed.squeeze().to("cpu")
        rep2 = rep2.squeeze().to("cpu")
        for pred in rep2:
            predictions_attribute_rep.append(pred.detach().numpy())
        for pred in composed:
            predictions_final_rep.append(pred.detach().numpy())
        loss_att = get_loss_cosine_distance(composed_phrase=rep2, original_phrase=batch["l"])
        loss_final = get_loss_cosine_distance(composed_phrase=composed, original_phrase=batch["l"])
        test_loss_att.append(loss_att.item())
        test_loss_final.append(loss_final.item())
        orig_phrases.append(batch["phrase"])
    orig_phrases = [item for sublist in orig_phrases for item in sublist]
    predictions_final_rep = np.array(predictions_final_rep)
    predictions_attribute_rep = np.array(predictions_attribute_rep)
    return predictions_final_rep, predictions_attribute_rep, np.average(test_loss_final), np.average(
        test_loss_att), orig_phrases


def save_predictions(predictions, path):
    """Save the predicted representations as numpy arrays"""
    np.save(file=path, arr=predictions, allow_pickle=True)


def get_save_path(model_path, split, representation_type):
    """
    Construct two paths (as strings) to save the predictions and ranks to
    :param model path: the base model path
    :param split: either dev or test
    :param representation_type: either "final_rep" or "attribute_rep"
    """
    prediction_path = model_path + representation_type + "_" + split + "predictions.npy"
    rank_path = model_path + representation_type + "_" + split + "ranks.npy"
    return prediction_path, rank_path


def evaluate(predictions_final_phrase, predictions_att_rep, ranks_final_phrase, ranks_att_rep, dataset, labels, config):
    """
    Load nearest neighbour ranker for each representation type and log the corresponding results
    :param predictions_final_phrase: the predicted representations (the final composed phrase)
    :param predictions_att_rep: the predicted representations (the attribute-specific representation)
    :param ranks_final_phrase: the path to save the ranks to (for final phrase)
    :param ranks_att_rep: the path to save the ranks to (for the attribute)
    :param dataset: the dataset to evaluate on
    :param labels: all possible labels that can be predicted
    :param config: the configuration that corresponds to the data
    """
    rank_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), num_workers=0,
                                              shuffle=False)
    ranker_attribute = NearestNeigbourRanker(path_to_predictions=predictions_att_rep,
                                             embedding_path=config["feature_extractor"]["static"][
                                                 "pretrained_model"], data_loader=rank_loader,
                                             all_labels=labels,
                                             y_label="phrase", max_rank=1000)
    ranker_attribute.save_ranks(ranks_att_rep)
    ranker_final_rep = NearestNeigbourRanker(path_to_predictions=predictions_final_phrase,
                                             embedding_path=config["feature_extractor"]["static"][
                                                 "pretrained_model"], data_loader=rank_loader,
                                             all_labels=labels,
                                             y_label="phrase", max_rank=1000)
    ranker_final_rep.save_ranks(ranks_final_phrase)

    logger.info(("result for learned attribute representation"))
    logger.info(ranker_attribute.result)
    logger.info("quartiles : %s" % str(ranker_attribute.quartiles))
    logger.info(
        "precision at rank 1: %.2f; precision at rank 5 %.2f" % (ranker_attribute._map_1, ranker_attribute._map_5))
    logger.info("accuracy: %.2f; f1 score: %.2f" % (ranker_attribute.accuracy, ranker_attribute.f1))

    logger.info(("\nresult for final representation"))
    logger.info(ranker_final_rep.result)
    logger.info("quartiles : %s" % str(ranker_final_rep.quartiles))
    logger.info(
        "precision at rank 1: %.2f; precision at rank 5 %.2f" % (ranker_final_rep._map_1, ranker_final_rep._map_5))
    logger.info("accuracy: %.2f; f1 score: %.2f" % (ranker_final_rep.accuracy, ranker_final_rep.f1))


if __name__ == "__main__":

    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config_1")
    argp.add_argument("path_to_config_2")
    argp.add_argument("--pretrain_dataset_1", default=False, action='store_true')
    argp.add_argument("--pretrain_dataset_2", default=False, action='store_true')
    argp = argp.parse_args()

    with open(argp.path_to_config_1, 'r') as f:
        config_1 = json.load(f)
    with open(argp.path_to_config_2, 'r') as f:
        config_2 = json.load(f)

    ts = time.gmtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # all save paths are picked based on the first configuration
    if config_1["save_name"] == "":
        save_name = format(
            "%s_%s" % (config_1["model"]["type"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))
    else:
        save_name = format("%s_%s" % (config_1["save_name"], time.strftime("%Y-%m-%d-%H_%M_%S", ts)))
    log_file = str(Path(config_1["logging_path"]).joinpath(save_name + "_log.txt"))
    model_path = str(Path(config_1["model_path"]).joinpath(save_name))

    prediction_path_dev_final_rep, rank_path_dev_final_rep = get_save_path(model_path, "_final_rep", "dev")
    prediction_path_test_final_rep, rank_path_test_final_rep = get_save_path(model_path, "_final_rep", "test")

    prediction_path_dev_attribute_rep, rank_path_dev_attribute_rep = get_save_path(model_path, "_attribute_rep", "dev")
    prediction_path_test_attribute_rep, rank_path_test_attribute_rep = get_save_path(model_path, "_attribute_rep",
                                                                                     "test")

    logging.config.dictConfig(create_config(log_file))
    logger = logging.getLogger("train")
    logger.info("Training a joint model with the following parameter for the first dataset: %s" % str(config_1))
    logger.info("The following parameter for the second dataset: %s" % str(config_2))

    # set random seed
    np.random.seed(config_1["seed"])
    # create two PretrainCompModel datasets
    dataset_train_1, dataset_valid_1, dataset_test_1 = get_datasets(config_1)
    dataset_train_2, dataset_valid_2, dataset_test_2 = get_datasets(config_2)

    assert type(dataset_train_1) == PretrainCompmodelDataset and type(
        dataset_train_2) == PretrainCompmodelDataset, "the dataset type is invalid for this kind of training"

    labels_dataset_1 = extract_all_words(training_data=config_1["train_data_path"],
                                         validation_data=config_1["validation_data_path"],
                                         test_data=config_1["test_data_path"],
                                         separator=config_1["data_loader"]["separator"],
                                         modifier=config_1["data_loader"]["modifier"],
                                         head=config_1["data_loader"]["head"],
                                         phrase=config_1["data_loader"]["phrase"])
    labels_dataset_2 = extract_all_labels(training_data=config_2["train_data_path"],
                                          validation_data=config_2["validation_data_path"],
                                          test_data=config_2["test_data_path"],
                                          separator=config_2["data_loader"]["separator"]
                                          , label=config_2["data_loader"]["phrase"])

    # load data with torch Data Loader
    combined_dataset = MultiRankingDataset(dataset_train_1, dataset_train_2)
    train_loader = DataLoader(combined_dataset,
                              batch_size=config_1["iterator"]["batch_size"],
                              shuffle=True,
                              num_workers=0)
    # load validation data in batches
    valid_loader_1 = torch.utils.data.DataLoader(dataset_valid_1,
                                                 batch_size=config_1["iterator"]["batch_size"],
                                                 shuffle=False,
                                                 num_workers=0)
    valid_loader_2 = torch.utils.data.DataLoader(dataset_valid_2, batch_size=len(dataset_valid_2), num_workers=0,
                                                 shuffle=False)

    model = None
    y_label = None
    # train
    train(config=config_1, pretrain_1=argp.pretrain_dataset_1, pretrain_2=argp.pretrain_dataset_2,
          train_loader=train_loader, valid_loader_1=valid_loader_1,
          valid_loader_2=valid_loader_2, device=device, model_path=model_path)
    # test and & evaluate
    logger.info("Loading best model from %s", model_path)
    valid_model = init_classifier(config_1)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()

    if valid_model:
        logger.info("generating predictions for validation data...")
        valid_predictions_final_phrase, valid_predictions_att, valid_loss_final, valid_loss_att, valid_phrases = \
            predict(
                valid_loader_2,
                valid_model, device)
        save_predictions(predictions=valid_predictions_final_phrase, path=prediction_path_dev_final_rep)
        save_predictions(predictions=valid_predictions_att, path=prediction_path_dev_attribute_rep)
        logger.info("saved predictions to %s" % prediction_path_dev_final_rep)
        logger.info("saved predictions to %s" % prediction_path_dev_attribute_rep)
        logger.info("validation loss att: %.5f" % valid_loss_att)
        logger.info("validation loss final phrase: %.5f" % valid_loss_final)
        evaluate(predictions_final_phrase=prediction_path_dev_final_rep,
                 predictions_att_rep=prediction_path_dev_attribute_rep, ranks_final_phrase=rank_path_dev_final_rep,
                 ranks_att_rep=rank_path_dev_attribute_rep, dataset=dataset_valid_2, labels=labels_dataset_2,
                 config=config_2)
    # do evaluation on test set if argument set to true
    if config_1["eval_on_test"]:
        logger.info("generating predictions for test data...")
        test_loader = torch.utils.data.DataLoader(dataset_test_2, batch_size=len(dataset_test_2), num_workers=0,
                                                  shuffle=False)
        test_predictions_final_phrase, test_predictions_att, test_loss_final, test_loss_att, test_phrases = predict(
            test_loader, valid_model, device)
        save_predictions(predictions=test_predictions_final_phrase, path=prediction_path_test_final_rep)
        save_predictions(predictions=test_predictions_att, path=prediction_path_test_attribute_rep)
        logger.info("saved predictions to %s" % prediction_path_test_final_rep)
        logger.info("saved predictions to %s" % prediction_path_test_attribute_rep)
        logger.info("validation loss att: %.5f" % test_loss_att)
        logger.info("validation loss final phrase: %.5f" % test_loss_final)
        evaluate(predictions_final_phrase=prediction_path_test_final_rep,
                 predictions_att_rep=prediction_path_test_attribute_rep, ranks_final_phrase=rank_path_test_final_rep,
                 ranks_att_rep=rank_path_test_attribute_rep, dataset=dataset_test_2, labels=labels_dataset_2,
                 config=config_2)

import argparse
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from utils import BertExtractor, StaticEmbeddingExtractor
from utils.training_utils import get_datasets
from training_scripts.nearest_neighbour import NearestNeigbourRanker
from utils.data_loader import extract_all_labels
import numpy as np
import pandas as pd

def get_average_phrase_static(data, feature_extractor):
    mods = data['modifier']
    heads = data['head']
    print(len(mods))
    print(len(heads))
    averages = []
    for m, h in zip(mods, heads):
        word1 = feature_extractor.get_embedding(m)
        word2 = feature_extractor.get_embedding(h)
        averages.append(np.sum((word1, word2), axis= 0))
    return averages


def get_average_phrase_BERT(data, feature_extractor):
    data = next(iter(data))
    mods = data['modifier']
    heads = data['head']
    sentences = data['context']
    averages = []
    for m, h,s in zip(mods, heads, sentences):

        word1 = feature_extractor.get_single_word_representations(s,m) #not sure
        word2 = feature_extractor.get_single_word_representations(s,h)
        averages.append(np.sum(word1, word2))
    return averages, data['label']


def save_predictions(predictions, path):
    np.save(file=path, arr=predictions, allow_pickle=True)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("config", help="the phrase dataset with context and labels")

    argp = argp.parse_args()
    # config mit static or contextualised
    with open(argp.config, 'r') as f:
        config = json.load(f)

    prediction_path = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_average_phrases.npy"))
    prediction_path_test = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_average_phrases_test.npy"))
    evaluation_path = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_results_val.npy"))
    evaluation_path_test = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_results_test.npy"))

    data_val = pd.read_csv(config["validation_data_path"], delimiter=config["data"]["separator"], index_col=False)
    data_test = pd.read_csv(config["test_data_path"],delimiter=config["data"]["separator"], index_col=False)

    print("TYPE", type(data_val), data_val.keys())

    if config["feature_extractor"]["contextualized_embeddings"] is False:
        feature_extractor = StaticEmbeddingExtractor(
            path_to_embeddings=config["feature_extractor"]["static"]["pretrained_model"])
        average_phrases= get_average_phrase_static(data_val, feature_extractor)
    else:
        bert_parameter = config["feature_extractor"]["contextualized"]["bert"]
        bert_model = bert_parameter["model"]
        max_len = bert_parameter["max_sent_len"]
        lower_case = bert_parameter["lower_case"]
        batch_size = bert_parameter["batch_size"]
        feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                         batch_size=batch_size)
        average_phrases = get_average_phrase_BERT(data_val, feature_extractor) #sentences will be needed here to get embeddings

    save_predictions(predictions=average_phrases, path=prediction_path)

    labels = extract_all_labels(training_data=config["train_data_path"],
                                validation_data=config["validation_data_path"],
                                test_data=config["test_data_path"],
                                separator=config["data"]["separator"]
                                , label=config["data"]["label"])

    ranker_attribute = NearestNeigbourRanker(path_to_predictions=prediction_path,
                                             embedding_extractor=feature_extractor,
                                             data_loader=[data_val],
                                             all_labels=labels,
                                             y_label="label", max_rank=1000)
    ranker_attribute.save_ranks(evaluation_path)

    if config["eval_on_test"]:
        if config["feature_extractor"]["contextualized_embeddings"] is False:
            average_phrases_test, labels_test = get_average_phrase_static(data_test, feature_extractor)
        else:
            average_phrases_test, labels_test = get_average_phrase_BERT(data_test, feature_extractor)
        ranker_attribute_test = NearestNeigbourRanker(path_to_predictions=prediction_path_test,
                                                      embedding_extractor=feature_extractor,
                                                      data_loader=[data_test],
                                                      all_labels=labels,
                                                      y_label="label", max_rank=1000)
        ranker_attribute_test.save_ranks(evaluation_path_test)
        """
        
        logger.info("result for learned attribute representation")
        logger.info(ranker_attribute.result)
        logger.info("quartiles : %s" % str(ranker_attribute.quartiles))
        logger.info(
            "precision at rank 1: %.2f; precision at rank 5 %.2f" % (ranker_attribute._map_1, ranker_attribute._map_5))
        logger.info("accuracy: %.2f; f1 score: %.2f" % (ranker_attribute.accuracy, ranker_attribute.f1))

        logger.info("saved ranks to %s" % rank_path_dev)
        """
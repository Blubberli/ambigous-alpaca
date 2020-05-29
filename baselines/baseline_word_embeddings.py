import argparse
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from utils import BertExtractor, StaticEmbeddingExtractor
from training_scripts.nearest_neighbour import NearestNeigbourRanker
from utils.data_loader import extract_all_labels, StaticRankingDataset, ContextualizedRankingDataset
import numpy as np


def get_average_phrase(data):
    """
    :param data: dataset as dataloader
    :return averages: list of averages of phrases
    """
    data_all = next(iter(data))
    averages = []
    for m, h in zip(data_all["w1"], data_all["w2"]):
        averages.append(m.add(h))
    return averages


def save_predictions(predictions, path):
    """
    :param predictions: predictions to save, here they are the averages vectors
    :param path: path to where to save predictions
    """
    np.save(file=path, arr=predictions, allow_pickle=True)


def save_results(ranker, path):
    """
    :param ranker: an object of a NearestNeighbourRanker
    :param path: where to save results
    """
    with open(path, 'w') as file:
        file.write("result : {} \n".format(ranker.result))
        file.write("quartiles : {} \n".format(ranker.quartiles))
        file.write("precision at rank 1:  {:0.2f} \n".format(ranker._map_1))
        file.write("precision at rank 5:  {:0.2f} \n".format(ranker._map_5))
        file.write("accuracy: {:0.2f}; f1 score: {:0.2f}".format(ranker.accuracy, ranker.f1))


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("config", help="has to be a specific config for baselines")
    argp = argp.parse_args()

    # config mit static or contextualised
    with open(argp.config, 'r') as f:
        config = json.load(f)

    prediction_path_val = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_average_phrases_val.npy"))
    prediction_path_test = str(
        Path(config["directory_path"]).joinpath(config["save_name"] + "_average_phrases_test.npy"))
    evaluation_path_val = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_results_val.npy"))
    evaluation_path_test = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_results_test.npy"))
    scores_path_val = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_scores_val.txt"))
    scores_path_test = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_scores_test.txt"))

    # get static embedding extractor and dataset if contextualized embeddings is false
    if config["feature_extractor"]["contextualized_embeddings"] is False:
        data_val = StaticRankingDataset(config["validation_data_path"],
                                        config["feature_extractor"]["static"]["pretrained_model"],
                                        config["data"]["separator"], config["data"]["modifier"],
                                        config["data"]["head"], config["data"]["label"])
        feature_extractor = StaticEmbeddingExtractor(
            path_to_embeddings=config["feature_extractor"]["static"]["pretrained_model"])
    # else get contextualised feature extractor and dataset
    else:
        bert_parameter = config["feature_extractor"]["contextualized"]["bert"]
        bert_model = bert_parameter["model"]
        max_len = bert_parameter["max_sent_len"]
        lower_case = bert_parameter["lower_case"]
        batch_size = bert_parameter["batch_size"]

        data_val = ContextualizedRankingDataset(config["validation_data_path"], bert_model, max_len,
                                                lower_case, batch_size, config["data"]["separator"],
                                                config["data"]["modifier"],
                                                config["data"]["head"], config["data"]["label"],
                                                config["label_definition_path"], context=config["data"]["context"])
        feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                          batch_size=batch_size)

    data_loader_val = DataLoader(data_val, batch_size=len(data_val), num_workers=0)
    average_phrases = get_average_phrase(data_loader_val)
    save_predictions(predictions=average_phrases, path=prediction_path_val)

    labels = extract_all_labels(training_data=config["train_data_path"],
                                validation_data=config["validation_data_path"],
                                test_data=config["test_data_path"],
                                separator=config["data"]["separator"]
                                , label=config["data"]["label"])

    ranker_attribute_val = NearestNeigbourRanker(path_to_predictions=prediction_path_val,
                                                 embedding_extractor=feature_extractor,
                                                 data_loader=data_loader_val,
                                                 all_labels=labels,
                                                 y_label="label", max_rank=1000)
    ranker_attribute_val.save_ranks(evaluation_path_val)
    save_results(ranker_attribute_val, scores_path_val)
    if config["eval_on_test"]:
        data_test = StaticRankingDataset(config["validation_data_path"],
                                         config["feature_extractor"]["static"]["pretrained_model"],
                                         config["data"]["separator"], config["data"]["modifier"],
                                         config["data"]["head"], config["data"]["label"])
        data_loader_test = DataLoader(data_val, batch_size=len(data_val), num_workers=0)
        if config["feature_extractor"]["contextualized_embeddings"] is False:
            average_phrases_test, labels_test = get_average_phrase(data_loader_test)
        else:
            average_phrases_test, labels_test = get_average_phrase(data_loader_test)
        ranker_attribute_test = NearestNeigbourRanker(path_to_predictions=prediction_path_test,
                                                      embedding_extractor=feature_extractor,
                                                      data_loader=[data_test],
                                                      all_labels=labels,
                                                      y_label=config["data"]["label"], max_rank=1000)
        ranker_attribute_test.save_ranks(evaluation_path_test)

        save_results(ranker_attribute_test, scores_path_test)

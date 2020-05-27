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
    print(data.keys())
    in_phrase = data['in_phrase']
    sentences = data['context']
    averages = []
    for phrase, sentence in zip(in_phrase, sentences):
        phrase_spl = phrase.split(" ")
        modifier = phrase_spl[0].strip()
        head = phrase_spl[1].strip()
        print(modifier, head, sentence)
        word1 = feature_extractor.get_single_word_representations([sentence], [modifier])
        word2 = feature_extractor.get_single_word_representations([sentence], [head])
        averages.append(np.sum(word1, word2), axis= 0)
    return averages, data['label']


def save_predictions(predictions, path):
    np.save(file=path, arr=predictions, allow_pickle=True)

def save_results(ranker, path):
    with open(path, 'w') as file:
        file.write("result : {} \n".format(ranker.result))
        file.write("quartiles : {} \n".format(ranker.quartiles))
        file.write("precision at rank 1:  {:0.2f} \n".format(ranker._map_1))
        file.write("precision at rank 5:  {:0.2f} \n".format(ranker._map_5))
        file.write("accuracy: {:0.2f}; f1 score: {:0.2f}".format(ranker.accuracy, ranker.f1))

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("config", help="the phrase dataset with context and labels")

    argp = argp.parse_args()
    # config mit static or contextualised
    with open(argp.config, 'r') as f:
        config = json.load(f)

    prediction_path_val = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_average_phrases_val.npy"))
    prediction_path_test = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_average_phrases_test.npy"))
    evaluation_path_val = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_results_val.npy"))
    evaluation_path_test = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_results_test.npy"))
    scores_path_val = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_scores_val.txt"))
    scores_path_test = str(Path(config["directory_path"]).joinpath(config["save_name"] + "_scores_test.txt"))

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

    save_predictions(predictions=average_phrases, path=prediction_path_val)

    labels = extract_all_labels(training_data=config["train_data_path"],
                                validation_data=config["validation_data_path"],
                                test_data=config["test_data_path"],
                                separator=config["data"]["separator"]
                                , label=config["data"]["label"])

    ranker_attribute_val = NearestNeigbourRanker(path_to_predictions=prediction_path_val,
                                             embedding_extractor=feature_extractor,
                                             data_loader=[data_val],
                                             all_labels=labels,
                                             y_label="label", max_rank=1000)
    ranker_attribute_val.save_ranks(evaluation_path_val)
    save_results(ranker_attribute_val, scores_path_val)
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

        save_results(ranker_attribute_test, scores_path_test)

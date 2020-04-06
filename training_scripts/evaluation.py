import argparse
import json
import numpy as np
from torch.utils.data import DataLoader
from training_scripts import Ranker
from utils.training_utils import get_datasets
from collections import defaultdict
from utils.data_loader import extract_all_labels


def class_performance_ranks(ranker, eval_path):
    """
    This method computes different evaluation statistics for a validation / testset ranker object and writes them to
    file. For each relation that occurs in the test set, the follwing statistics will be computed:
    - quartiles: (average rank of 25, 50 and 75 percent of the per-relation ranks
    - percentages: percentage of per-relation ranks == 1 and <= 5
    - 5 most similar relations (based on cosine similarity) to the original relation representation
    - cosine similarities to similar relations
    - cosine similarity between composed and gold relation representation
    :param ranker: a ranking object that has been created for a validation or test set
    :param eval_path: the path where the statistics will be written to
    """
    relation2gold_similarities = defaultdict(list)
    relation2comp_similarities = defaultdict(list)
    relation2ranks = defaultdict(list)
    index2label = dict(zip(list(ranker.label2index.values()), list(ranker.label2index.keys())))

    for i in range(len(ranker.true_labels)):
        relation2gold_similarities[ranker.true_labels[i]].append(ranker.gold_similarities[i])
        relation2comp_similarities[ranker.true_labels[i]].append(ranker.composed_similarities[i])
        relation2ranks[ranker.true_labels[i]].append(ranker.ranks[i])
    for rel, sims in relation2gold_similarities.items():
        sims = np.average(np.array(sims).transpose(), axis=1)
        most_similar_indices = sims.argsort()[-5:][::-1]
        most_similar_relations = [index2label[index] for index in most_similar_indices]
        most_similar_sims = [np.round(sims[index], decimals=2) for index in most_similar_indices]

        composed2rel_sim = np.average(relation2comp_similarities[rel])
        quartiles, p = Ranker.calculate_quartiles(relation2ranks[rel])
        s = "\nrelation : %s\nquartiles: %s\npercentages: %s\nmost similar relations to gold rel: %s\ncosine " \
            "similarities to similar relations : %s, average cosine similarity composed vs gold : %.2f" % (
                rel, str(quartiles), p, str(most_similar_relations), str(most_similar_sims), composed2rel_sim)
        with open(eval_path, "w") as f:
            f.write(s)


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp.add_argument('--ranking', default=False, action='store_true')
    argp = argp.parse_args()

    with open(argp.path_to_config, 'r') as f:
        config = json.load(f)
    prediction_path_dev = config["pretrained_model_path"] + "_dev_predictions.npy"
    prediction_path_test = config["pretrained_model_path"] + "_test_predictions.npy"
    eval_path_dev = config["pretrained_model_path"] + "_evaluation_dev.txt"
    eval_path_test = config["pretrained_model_path"] + "_evaluation_test.txt"
    dataset_train, dataset_valid, dataset_test = get_datasets(config)

    # load validation data in batches
    valid_loader = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    # load test data in batches
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    if argp.ranking:
        rank_path_dev = config["pretrained_model_path"] + "_dev_ranks.txt"
        rank_path_test = config["pretrained_model_path"] + "_test_ranks.txt"
        labels = extract_all_labels(training_data=config["train_data_path"],
                                    validation_data=config["validation_data_path"],
                                    test_data=config["test_data_path"],
                                    separator=config["data_loader"]["separator"]
                                    , label=config["data_loader"]["phrase"])
        ranker = Ranker(path_to_predictions=prediction_path_dev, embedding_path=config["feature_extractor"]["static"][
            "pretrained_model"], data_loader=valid_loader, all_labels=labels,
                        y_label="phrase", max_rank=1000)
        class_performance_ranks(ranker, eval_path_dev)
        if config["eval_on_test"]:
            ranker = Ranker(path_to_predictions=prediction_path_test,
                            embedding_path=config["feature_extractor"]["static"][
                                "pretrained_model"], data_loader=test_loader, all_labels=labels, y_label="phrase",
                            max_rank=1000)
            class_performance_ranks(ranker, eval_path_test)

    else:
        print("code for classification")

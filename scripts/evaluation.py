import argparse
import json
import numpy as np
from torch.utils.data import DataLoader
from scripts.ranking import Ranker
from scripts.training_utils import get_datasets
from collections import defaultdict
from scripts.data_loader import extract_all_labels
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


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
    eval_file = open(eval_path, "w")
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

        eval_file.write(s)
    eval_file.close()


def class_performance_classification(path_predictions, gold_loader, dataset, eval_path):
    """
    this method calculates precision,recall and f1 for each class separately.
    the results are written to a file
    :param path_predictions: path to saved predictions
    :param gold_loader: the dataloader to generate the gold labels
    :param dataset: the dataset object to call the encoder of the labels
    :param eval_path: the path of the file to which to save the scores
    """
    preds = np.load(path_predictions)
    gold = next(iter(gold_loader))
    gold = gold["l"].numpy()
    set_labels = set(gold)
    report = classification_report(gold, preds, labels=list(set_labels),  output_dict=True)
    encoder = dataset.label_encoder
    f = open(eval_path, "w")
    for k, v in report.items():
        if k.isdigit():
            label = encoder.inverse_transform([int(k)])
            scores = {type_score:round(score, 3) for type_score, score in v.items()}
            f.write(str(label) + "\t" + str(scores) + "\n")
        else:
            scores = {type_score: round(score, 3) for type_score, score in v.items()}
            f.write(str(k) + "\t" + str(scores) + "\n")

    f.close()


def plot_confusion_matrix(confusion_matrix, save_path):
    """
    Method to create a plot of a confusion matrix
    :param confusion_matrix: the confusion matrix as a dataframe
    :param save_path: the path to save the plot as png
    """
    plt.figure(figsize=(10, 11))
    sn.heatmap(confusion_matrix, annot=True)
    sn.set(font_scale=0.9)
    plt.savefig(save_path, dpi=300)


def confusion_matrix_classification(path_predictions, gold_loader, dataset, conf_path, plot):
    """
    this method generates a confusion matrix for the predicted and gold labels
    :param path_predictions: path to saved predictions
    :param gold_loader: the dataloader to generate the gold labels
    :param dataset: the dataset object to call the encoder of the labels
    :param conf_path: path to which to save the confusion matrix
    :param plot: if true a plot of the confusion matrix is also saved
    """
    preds = np.load(path_predictions)
    gold = next(iter(gold_loader))
    gold = gold["l"].numpy()
    encoder = dataset.label_encoder
    gold_l = encoder.inverse_transform(gold)
    preds_l = encoder.inverse_transform(preds)
    labels = np.unique(np.concatenate((preds_l, gold_l), axis=0))
    conf_matrix = confusion_matrix(gold_l, preds_l)
    conf_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    if plot:
        plot_confusion_matrix(confusion_matrix=conf_matrix, save_path=conf_path.replace("csv", "png"))
    conf_matrix.to_csv(conf_path, sep="\t")


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp.add_argument('--confusion_matrix', default=False, action='store_true')
    argp.add_argument("--plot_matrix", default=False, action='store_true')
    argp.add_argument('--ranking', default=False, action='store_true')
    argp = argp.parse_args()

    with open(argp.path_to_config, 'r') as f:
        config = json.load(f)
    prediction_path_dev = str(Path(config["model_path"]).joinpath(config["save_name"] + "_dev_predictions.npy"))
    prediction_path_test = str(Path(config["model_path"]).joinpath(config["save_name"] + "_test_predictions.npy"))
    eval_path_dev = str(Path(config["model_path"]).joinpath(config["save_name"] + "_evaluation_dev.txt"))
    eval_path_test = str(Path(config["model_path"]).joinpath(config["save_name"] + "_evaluation_test.txt"))
    dataset_train, dataset_valid, dataset_test = get_datasets(config)

    # load validation data in batches
    valid_loader = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    # load test data in batches
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    if argp.ranking:
        rank_path_dev = config["model_path"] + "_dev_ranks.txt"
        rank_path_test = config["model_path"] + "_test_ranks.txt"
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

        if config["eval_on_test"]:
            class_performance_classification(path_predictions=prediction_path_dev, gold_loader=valid_loader, dataset=dataset_valid, eval_path=eval_path_dev)
            class_performance_classification(path_predictions=prediction_path_test, gold_loader=test_loader, dataset= dataset_test,eval_path= eval_path_test)
            if argp.confusion_matrix:
                conf_path_dev = str(Path(config["model_path"]).joinpath(config["save_name"] + "_confusion_dev.csv"))
                conf_path_test = str(Path(config["model_path"]).joinpath(config["save_name"] + "_confusion_test.csv"))
                confusion_matrix_classification(path_predictions=prediction_path_dev, gold_loader=valid_loader,
                                                dataset=dataset_valid, conf_path=conf_path_dev, plot=argp.plot_matrix)
                confusion_matrix_classification(path_predictions=prediction_path_test, gold_loader=test_loader,
                                                dataset=dataset_test, conf_path=conf_path_test, plot=argp.plot_matrix)

        else:
            class_performance_classification(path_predictions=prediction_path_dev, gold_loader=valid_loader, dataset=dataset_valid, eval_path=eval_path_dev)
            if argp.confusion_matrix:
                conf_path_dev = str(Path(config["model_path"]).joinpath(config["save_name"] + "_confusion_dev.csv"))
                confusion_matrix_classification(path_predictions=prediction_path_dev, gold_loader=valid_loader,
                                                dataset=dataset_valid, conf_path=conf_path_dev,
                                                plot=argp.plot_matrix)

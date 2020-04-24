import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import json
from torch.utils.data import DataLoader
from scripts import training_utils
import seaborn as sns
import pandas as pd


def plot_learning_curves(training_losses, validation_losses, save_path):
    """
    Given two lists of training losses and validation losses over a range of epochs, this method creates a plot to
    see the learning curves of the model
    :param training_losses: a list of training losses
    :param validation_losses: a list of validation losses
    :param save_path: the path the plot is saved to
    """
    assert len(training_losses) == len(
        validation_losses), "lists of train and validation losses have to have the same length"
    d = pd.DataFrame({"train loss": training_losses, "val loss": validation_losses})
    sns.lineplot(data=d, hue=["train loss", "val loss"])
    plt.savefig(save_path, dpi=400)


def plot_overall_class_distribution(labels, save_path):
    """
    Given a list of labels for one dataset, this method creates a histogram
    that shows the overall class distribution
    :param labels: list of labels
    :param save_path: path to save the plot to.
    """
    unique, counts = np.unique(labels, return_counts=True)
    tup = list(zip(unique, counts))
    tup.sort(key=lambda tup: tup[1])
    unique = [t[0] for t in tup]
    counts = [t[1] for t in tup]
    plt.bar(unique, counts, color='red', alpha=0.7)
    max1 = max(counts)

    train_dist = mpatches.Patch(color='red', alpha=0.7, label='classes in dataset')

    plt.legend(handles=[train_dist])
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(np.arange(0, max1, step=25))
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()


def plot_class_distribution_splits(train_labels, save_path, test_labels, test_label):
    """
    Given a list of training labels and a list of validation labels for a dataset, this method creates a histogram
    that shows the class distribution of each split in the same plot. The colors are transparent so that both
    distributions can be inspected at the same time
    :param train_labels: a list containing the label for each training instance
    :param test_labels: a list containing the label for each test instance
    :param save_path: the path the plot is saved to
    :param test_label: the name of the test split (e.g. test or validation) as a String
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    plt.bar(unique, counts, color='red', alpha=0.5)
    max1 = max(counts)
    unique, counts = np.unique(test_labels, return_counts=True)
    plt.bar(unique, counts, color='orange', alpha=0.5)
    max2 = max(counts)
    train_dist = mpatches.Patch(color='red', alpha=0.5, label='train')
    test_dist = mpatches.Patch(color='orange', alpha=0.5, label=test_label)

    plt.legend(handles=[train_dist, test_dist])
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(np.arange(0, max(max1, max2), step=25))
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config", type=str)
    argp.add_argument("save_path", type=str)
    argp.add_argument("--test_data", default=False, action='store_true')
    argp = argp.parse_args()
    with open(argp.path_to_config, 'r') as f:
        config = json.load(f)

    dataset_train, dataset_valid, dataset_test = training_utils.get_datasets(config)
    label_encoder = dataset_train.label_encoder
    # load training data
    train_loader = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    train_labels = next(iter(train_loader))["l"].numpy()
    train_labels = label_encoder.inverse_transform(train_labels)

    # load validation data in batches
    valid_loader = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    valid_labels = next(iter(valid_loader))["l"].numpy()
    valid_labels = label_encoder.inverse_transform(valid_labels)

    plot_class_distribution_splits(train_labels=train_labels, test_labels=valid_labels,
                                   save_path=argp.save_path + "_validation.png", test_label="validation")

    # load test data in batches
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    test_labels = next(iter(test_loader))["l"].numpy()
    test_labels = label_encoder.inverse_transform(test_labels)

    all_labels = list(train_labels) + list(valid_labels) + list(test_labels)
    plot_overall_class_distribution(all_labels, argp.save_path + "_overall_dataset.png")

    if argp.test_data:
        plot_class_distribution_splits(train_labels=train_labels, test_labels=test_labels,
                                       save_path=argp.save_path + "_test.png", test_label="test")

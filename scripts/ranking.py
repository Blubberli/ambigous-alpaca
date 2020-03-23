import math
import numpy as np
import torch
import torch.nn.functional as F
from scripts import StaticEmbeddingExtractor


class Ranker:

    def __init__(self, path_to_predictions, path_to_ranks, embedding_path, data_loader, all_labels, max_rank, y_label):
        """
        This class stores the functionality to rank a prediction with respect to some gold standard representations and
        to compute the precision at certain ranks or the quartiles
        :param path_to_predictions: [String] The path to numpy array stored predictions (number_of_test_instances x
        embedding_dim)
        :param path_to_ranks: [String] the path were the computed ranks will be saved to
        :param embedding_path: [String] the path to the embeddings
        :param data_loader: [Dataloader] a data loader witch batchsize 1 that holds the test data
        :param all_labels: [list(String)] a list of all unique labels that can occur in the test data
        :param max_rank: [int] the worst possible rank (even if an instance would get a lower rank it is set to this
        number)
        :param y_label: [String] the column name of the label in the test data
        """
        # load composed predictions
        self._predicted_embeddings = np.load(path_to_predictions, allow_pickle=True)
        self._embeddings = StaticEmbeddingExtractor(embedding_path)
        data = next(iter(data_loader))
        # the correct labels are stored here
        self._true_labels = data[y_label]
        self._max_rank = max_rank

        # construct label embedding matrix, embeddings of labels are looked up in the original embeddings
        all_labels = sorted(all_labels)
        self._label_embeddings = self._embeddings.get_array_embeddings(all_labels)
        self._label2index = dict(zip(all_labels, range(len(all_labels))))
        # normalize predictions and label embedding matrix (in case they are not normalized)
        self._label_embeddings = F.normalize(torch.from_numpy(np.array(self._label_embeddings)), p=2, dim=1)
        self._predicted_embeddings = F.normalize(torch.from_numpy(np.array(self._predicted_embeddings)), p=2, dim=1)
        # compute the ranks, quartiles and precision
        self._ranks = self.get_target_based_rank()
        self._quartiles, self._result = self.calculate_quartiles(self._ranks)
        # save the ranks
        self.save_ranks(self._ranks, path_to_ranks)

    def get_target_based_rank(self):
        """
        Computes the ranks of the composed representations, given a matrix of gold standard label embeddings.
        The ordering is relative to the gold standard target/label representation.
        :return: a list with the ranks for all the composed representations in the batch
        """
        all_ranks = []
        # get the index for each label in the true labels
        target_idxs = [self.label2index[label] for label in self.true_labels]

        # get a matrix, each row representing the gold representation of the corresponding label
        target_repr = np.take(self.label_embeddings, target_idxs, axis=0)

        # get the similarity between each label and each other possible label
        # result: [labelsize x targetinstances]  = for each instance a vector of cosine similarities to each label
        target_dict_similarities = np.dot(self.label_embeddings, np.transpose(target_repr))

        for i in range(self._predicted_embeddings.shape[0]):
            # compute similarity between the target and the predicted vector
            target_composed_similarity = np.dot(self.predicted_embeddings[i], target_repr[i])
            # delete the similarity between the target label and itself
            target_sims = np.delete(target_dict_similarities[:, i], target_idxs[i])

            # the rank is the number of vectors with greater similarity that the one between
            # the target representation and the composed one; no sorting is required, just
            # the number of elements that are more similar
            rank = np.count_nonzero(target_sims > target_composed_similarity) + 1
            if rank > self.max_rank:
                rank = self.max_rank
            all_ranks.append(rank)

        return all_ranks

    def save_ranks(self, ranks, file_to_save):
        with open(file_to_save, "w", encoding="utf8") as f:
            for i in range(len(self._true_labels)):
                f.write(self.true_labels[i] + " " + str(ranks[i]) + "\n")
        print("ranks saved to file: " + file_to_save)

    @staticmethod
    def calculate_quartiles(ranks):
        """
        get the quartiles for the data
        :param ranks: a list of ranks
        :return: the three quartiles we are interested in, string representation of percentage of data that are rank 1
        and percentage of data that are
        """
        sorted_data = sorted(ranks)
        leq5 = sum([1 for rank in sorted_data if rank <= 5])
        leq1 = sum([1 for rank in sorted_data if rank == 1])
        mid_index = math.floor((len(sorted_data) - 1) / 2)
        if len(sorted_data) % 2 != 0:
            quartiles = list(map(np.median, [sorted_data[0:mid_index], sorted_data, sorted_data[mid_index + 1:]]))
        else:
            quartiles = list(map(np.median, [sorted_data[0:mid_index + 1], sorted_data, sorted_data[mid_index + 1:]]))
        return quartiles, "%.2f of ranks = 1; %.2f%% of ranks <=5" % (
            (100 * leq1 / float(len(sorted_data))), (100 * leq5 / float(len(sorted_data))))

    @property
    def predicted_embeddings(self):
        return self._predicted_embeddings

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def true_labels(self):
        return self._true_labels

    @property
    def max_rank(self):
        return self._max_rank

    @property
    def label_embeddings(self):
        return self._label_embeddings

    @property
    def label2index(self):
        return self._label2index

    @property
    def ranks(self):
        return self._ranks

    @property
    def quartiles(self):
        return self._quartiles

    @property
    def result(self):
        return self._result

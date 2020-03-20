import math
import numpy as np
import torch
import torch.nn.functional as F
from scripts import StaticEmbeddingExtractor


class Ranker:

    def __init__(self, path_to_predictions, path_to_ranks, embedding_path, dataloader, all_labels, max_rank):
        # load composed predictions
        # I have a dataset, e.g. test data. i can iterate through it and for each batch I have:
        # the original phrases
        # I need the matrix of label embeddings and I need to know which embedding belongs to which label
        self._predicted_embeddings = np.load(path_to_predictions, allow_pickle=True)
        self._embeddings = StaticEmbeddingExtractor(embedding_path)
        data = next(iter(dataloader))
        # the correct labels are stored here
        self._true_labels = data["phrase"]
        self._max_rank = max_rank

        # construct label embedding matrix
        all_labels = sorted(all_labels)
        self._label_embeddings = self._embeddings.get_array_embeddings(all_labels)
        self._label2index = dict(zip(all_labels, range(len(all_labels))))
        # normalize embeddings
        self._label_embeddings = F.normalize(torch.from_numpy(np.array(self._label_embeddings)), p=2, dim=1)
        self._ranks = self.get_target_based_rank()
        print(self.calculate_quartiles(self._ranks))
        self.save_ranks(self._ranks, path_to_ranks)

    def get_target_based_rank(self):
        """
        Computes the ranks of the composed representations, given a dictionary of embeddings.
        The ordering is relative to the target representation.
        :param composed_repr: a batch of composed representations
        :param targets: a batch of targets (i.e. the phrases to be composed)
        :param max_rank: the maximum rank
        :param dictionary_embeddings: a gensim model containing the original embeddings
        :return: a list with the ranks for all the composed representations in the batch
        """
        # self._label_embeddings = F.normalize(torch.from_numpy(np.array(self._label_embeddings)), p=2, dim=1)
        # self._predicted_embeddings = F.normalize(torch.from_numpy(np.array(self._predicted_embeddings)), p=2, dim=1)
        all_ranks = []
        # get the index for each label in the true labels
        target_idxs = [self._label2index[label] for label in self._true_labels]
        # get a matrix, each row representing the gold representation of the corresponding label
        target_repr = np.take(self._label_embeddings, target_idxs, axis=0)

        # get the similarity between each label and each other possible label
        # result: [labelsize x targetinstances]  = for each instance a vector of cosine similarities to each label
        target_dict_similarities = np.dot(self._label_embeddings, np.transpose(target_repr))

        for i in range(self._predicted_embeddings.shape[0]):
            # compute similarity between the target and the predicted vector
            target_composed_similarity = np.dot(self._predicted_embeddings[i], target_repr[i])

            target_sims = np.delete(target_dict_similarities[:, i], target_idxs[i])

            # the rank is the number of vectors with greater similarity that the one between
            # the target representation and the composed one; no sorting is required, just
            # the number of elements that are more similar
            rank = np.count_nonzero(target_sims > target_composed_similarity) + 1
            if (rank > self._max_rank):
                rank = self._max_rank
            all_ranks.append(rank)

        return all_ranks

    def save_ranks(self, ranks, file_to_save):
        with open(file_to_save, "w", encoding="utf8") as f:
            for i in range(len(self._true_labels)):
                f.write(self._true_labels[i] + " " + str(ranks[i]) + "\n")
        print("ranks saved to file: " + file_to_save)

    @staticmethod
    def calculate_quartiles(ranks):
        """
        get the quartiles for the data
        :param ranks: a list of ranks
        :return: the three quartiles we are interested in
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

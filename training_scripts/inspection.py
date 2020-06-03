import torch
from utils.training_utils import get_datasets, init_classifier
from utils import StaticRankingDataset
from utils.loss_functions import get_loss_cosine_distance
from utils.data_loader import extract_all_labels
import argparse
import json
from training_scripts.train_simple_ranking import save_predictions
from training_scripts.nearest_neighbour import NearestNeigbourRanker
# from training_scripts.train_simple_ranking import save_predictions
from utils import StaticEmbeddingExtractor, BertExtractor
import numpy
from training_scripts.evaluation import class_performance_nearest_neighbour, confusion_matrix_ranking
import collections


def predict_simple_composition_model(model_path, test_data_loader, training_config, save_path):
    """
    Given a trained model and a new dataset, predict the adj-noun phrase representations with the trained model
    :param model_path: path to a model that has been trained on some data
    :param test_data_loader: dataloader for test data with modifier and head embeddings
    :param training_config: config of the corresponding trained model
    :param save_path: path were the predictions will be saved to (with the ending "attribute_predictions.npy")
    :return: predictions for the given dataset and the average similarity between the correct attribute and the
    predicted vector
    """
    valid_model = init_classifier(training_config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()
    data = next(iter(test_data_loader))
    data["device"] = "cpu"
    if valid_model:
        valid_model.load_state_dict(torch.load(model_path))
        valid_model.eval()
        composed = valid_model(data)
        composed = composed.squeeze().to("cpu").detach().numpy()
        distance_composed_true_att = get_loss_cosine_distance(composed_phrase=composed, original_phrase=data["l"])
        average_similarity = 1 - distance_composed_true_att.item()
        save_predictions(predictions=composed, path=save_path + "attribute_predictions.npy")
        return composed, average_similarity


def predict_joint_composition_model(model_path, test_data_loader, training_config, save_path):
    """
    Given a trained model and a new dataset, predict the adj-noun phrase representations with the trained model
    :param model_path: path to a model that has been trained on some data
    :param test_data_loader: dataloader for test data with modifier and head embeddings
    :param training_config: config of the corresponding trained model
    :param save_path: path were the predictions will be saved to (with the ending "attribute_predictions.npy",
    "reconstructed_predictions.npy", and "combined_predictions.npy")
    :return: a dictionary with similarities between the different representations and a dictionary with:
        - attribute : the attribute representations
        - reconstructed : the reconstructed representations
        - combined : the combination of these two (sum of attribute and reconstructed normalized to unit norm)
    """
    valid_model = init_classifier(training_config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()
    data = next(iter(test_data_loader))
    data["device"] = "cpu"
    representation_dic = {}
    if valid_model:
        valid_model.load_state_dict(torch.load(model_path))
        valid_model.eval()
        composed, rep1, rep2 = valid_model(data)
        combined = composed.squeeze().to("cpu")
        attribute = rep2.squeeze().to("cpu")
        reconstructed = rep1.squeeze().to("cpu")

        similarity_dic = get_average_similarities(reconstructed, attribute, combined, data)
        save_predictions(predictions=reconstructed.detach().numpy(), path=save_path + "reconstructed_predictions.npy")
        save_predictions(predictions=attribute.detach().numpy(), path=save_path + "attribute_predictions.npy")
        save_predictions(predictions=combined.detach().numpy(), path=save_path + "combined_predictions.npy")
        representation_dic["attribute"] = attribute.detach().numpy()
        representation_dic["reconstructed"] = reconstructed.detach().numpy()
        representation_dic["combined"] = combined.detach().numpy()
        return similarity_dic, representation_dic


def get_average_similarities(reconstructed, attribute, combined, data):
    """
    This method computes the cosine similaritiy between different representations and returns these similarities in a
    dictionary
    :param reconstructed: the representation that represents the reconstructed adj-noun vector
    :param attribute: the representation that represents the learned attribute vector
    :param combined: the combination of the two vectors, which is the sum of the vector normalized to unit length
    :param data: the test data that contains the corresponding correct attribute vector
    :return: a dictionary with the following similarities:
        - attribute2realAttribute: the average similarity between the learned attribute vector and the correct
        attribute vector
        - reconstructed2realAttribute: the average similarity between the learned reconstructed vector and the
        correct attribute vector
        - combined2realAttribute: the average similarity between the final combined representation and the correct
        attribute vector
        - attribute2reconstructed: the average similarity between the learned attribute vector and the reconstructed
        vector
    """
    similarity_dictionary = {}
    similarity_dictionary["attribute2realAttribute"] = 1 - get_loss_cosine_distance(composed_phrase=attribute,
                                                                                    original_phrase=data["l"]).item()
    similarity_dictionary["reconstructed2realAttribute"] = 1 - get_loss_cosine_distance(composed_phrase=reconstructed,
                                                                                        original_phrase=data[
                                                                                            "l"]).item()
    similarity_dictionary["combined2realAttribute"] = 1 - get_loss_cosine_distance(composed_phrase=combined,
                                                                                   original_phrase=data["l"]).item()
    similarity_dictionary["attribute2reconstructed"] = 1 - get_loss_cosine_distance(original_phrase=attribute,
                                                                                    composed_phrase=reconstructed).item()
    return similarity_dictionary


def get_nearest_neighbours_for_given_list(vector, vector_list, index2label):
    """
    Given a vector and a list of other vectors, each vectors index in that list corresponding to a real word that is
    defined with the dictionary "index2label", compute the similarities between the given vector and each word and
    return these in a sorted dictionary
    :param vector: a vector (e.g. a predicted representation for an adj-noun pair)
    :param vector_list: a list of vectors [vector1, vector2, vector2]
    :param index2label: a dictionary that maps the list of vectors to a word, e.g. if 0 = Haus, 1 is Liebe and 2 is
    Wahrheit, then vector1 = Haus, vector2 = Liebe, vector3 = Wahrheit
    :return: sorted List of tuples. Each tuple contains the word and the similarity that was computed between the
    given vector and that word. Sorted from most similar to lease similar words
    """
    vec2label_sim = numpy.dot(vector, vector_list.transpose())
    vec2label_sim = dict(zip([v for k, v in index2label.items()], vec2label_sim))
    vec2label_sim = sorted(vec2label_sim.items(), key=lambda kv: kv[1])
    vec2label_sim.reverse()
    return vec2label_sim


def nearest_neighbours_static(predicted_vectors, feature_extractor, dataset, all_labels, save_name):
    """
    For all predicted vectors for a given dataset extract the nearest neighbours from
    a) the whole embedding space
    b) a list of labels
    :param predicted_vectors: a numpy array of predicted representations for a given dataset
    :param feature_extractor: a static feature extractor
    :param dataset: a single dataset that contains modifier, head and label
    :param all_labels:
    :return:
    """
    modifier = dataset.modifier_words
    heads = dataset.head_words
    labels = dataset.phrases
    label_embeddings = numpy.array(feature_extractor.get_array_embeddings(all_labels))
    index2label = dict(zip(range(len(all_labels)), all_labels))
    f = open(save_name + "_nearest_neighbours.txt", "w")
    for i in range(predicted_vectors.shape[0]):
        vec = predicted_vectors[i]
        label = labels[i]
        phrase = modifier[i] + " " + heads[i]
        vec2label_sim = get_nearest_neighbours_for_given_list(vec, label_embeddings, index2label)
        top_labels = [el[0] for el in vec2label_sim]
        general_neighbours = feature_extractor.embeds.embedding_similarity(vec)
        s = "phrase: %s \n correct label: %s\n top predicted labels: %s \n general close words: %s\n" % (
            phrase, label, str(top_labels[:5]), str(general_neighbours[:5]))
        f.write(s)
    f.close()


# file to measure

def get_dataset(config, data_path):
    """
    This method creates a single Dataset with modifier, head and label (embeddings).
    :param config: a configuration that specified the format of the dataset
    :param data_path: a path to a dataset that should be used to create predictions for
    :return: a RankingDataset (either static or contextual)
    """
    separator = config["data_loader"]["separator"]
    mod = config["data_loader"]["modifier"]
    head = config["data_loader"]["head"]
    label = config["data_loader"]["label"]
    if config["feature_extractor"]["contextualized_embeddings"]:
        # a contextualizedrankingdataset will be contructed
        pass
    else:
        embedding_path = config["feature_extractor"]["static"]["pretrained_model"]
        dataset = PretrainCompmodelDataset(data_path=data_path,
                                           embedding_path=embedding_path, separator=separator,
                                           phrase=label, mod=mod, head=head)

    return dataset


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("test_data")
    argp.add_argument("training_config")
    argp.add_argument("model_path")
    argp.add_argument("save_name")
    argp.add_argument("--labels")
    argp = argp.parse_args()

    with open(argp.training_config, 'r') as f:
        training_config = json.load(f)

    prediction_path_dev = argp.save_name + "_dev_predictions.npy"
    rank_path_dev = argp.save_name + "_dev_ranks.txt"
    dataset = get_dataset(config=training_config, data_path=argp.test_data)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=len(dataset),
                                              shuffle=False,
                                              num_workers=0)
    if argp.labels:
        print("extract labels from file")
    else:
        labels = extract_all_labels(training_data=training_config["train_data_path"],
                                    validation_data=training_config["validation_data_path"],
                                    test_data=training_config["test_data_path"],
                                    separator=training_config["data_loader"]["separator"]
                                    , label=training_config["data_loader"]["phrase"])
    sim_dic, rep_dic = predict_joint_composition_model(model_path=argp.model_path, test_data_loader=data_loader,
                                                       training_config=training_config, save_path=argp.save_name)
    feature_extractor = StaticEmbeddingExtractor(
        path_to_embeddings=training_config["feature_extractor"]["static"]["pretrained_model"])
    # nearest_neighbours_static(rep_dic["reconstructed"], feature_extractor, dataset, labels,
    # argp.save_name+"_reconstructed")
    # nearest_neighbours_static(rep_dic["attribute"], feature_extractor, dataset, labels, argp.save_name+"_attribute")
    # nearest_neighbours_static(rep_dic["combined"], feature_extractor, dataset, labels, argp.save_name+"_combined")
    ranker = NearestNeigbourRanker(embedding_extractor=feature_extractor, all_labels=labels, data_loader=data_loader,
                                   path_to_predictions=argp.save_name + "combined_predictions.npy", y_label="phrase",
                                   max_rank=49)
    confusion_matrix_ranking(ranker=ranker, save_path=argp.save_name + "combined_predictions.npy")
    results = class_performance_nearest_neighbour(ranker, argp.save_name + "combined_predictions.npy")

import torch
from utils.training_utils import get_datasets, init_classifier
from utils.data_loader import extract_all_labels
import argparse
import json
from training_scripts.nearest_neighbour import NearestNeigbourRanker
from training_scripts.pretrain import save_predictions
from utils import StaticEmbeddingExtractor, BertExtractor


def predict_simple_composition_model(model_path, test_data_loader, training_config):
    """
    Given a trained model and a new dataset, predict the adj-noun phrase representations with the trained model
    :param model_path: path to a model that has been trained on some data
    :param test_data_loader: dataloader of the new data with modifier and head embeddings
    :param training_config: config of the corresponding trained model
    :return: predictions for the new dataset
    """
    valid_model = init_classifier(training_config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()
    data = next(iter(test_data_loader))
    if valid_model:
        valid_model.load_state_dict(torch.load(model_path))
        valid_model.eval()
        composed = valid_model(data)
        composed = composed.squeeze().to("cpu").detach().numpy()

        return composed


def evaluate(predictions, embedding_extractor, data_loader, rank_path):
    """
    Given a list of predictions for a new dataset, compute and print the nearest neighbour results
    :param predictions: a list of vectors, representing the predicted adj-noun phrases of the new dataset
    :param embedding_extractor: an embedding extractor
    :param data_loader: the data loader of the new dataset
    :param rank_path: the path to save the ranks to
    """
    ranker_attribute = NearestNeigbourRanker(path_to_predictions=predictions,
                                             embedding_extractor=embedding_extractor,
                                             data_loader=data_loader,
                                             all_labels=labels,
                                             y_label="phrase", max_rank=1000)
    ranker_attribute.save_ranks(rank_path)

    print(ranker_attribute.result)
    print("quartiles : %s" % str(ranker_attribute.quartiles))
    print(
        "precision at rank 1: %.2f; precision at rank 5 %.2f" % (ranker_attribute._map_1, ranker_attribute._map_5))
    print("accuracy: %.2f; f1 score: %.2f" % (ranker_attribute.accuracy, ranker_attribute.f1))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("test_config")
    argp.add_argument("training_config")
    argp.add_argument("model_path")
    argp.add_argument("save_name")
    argp = argp.parse_args()

    with open(argp.test_config, 'r') as f:
        test_config = json.load(f)
    with open(argp.training_config, 'r') as f:
        training_config = json.load(f)

    prediction_path_dev = argp.save_name + "_dev_predictions.npy"
    rank_path_dev = argp.save_name + "_dev_ranks.txt"
    dataset_train, dataset_valid, dataset_test = get_datasets(test_config)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=len(dataset_valid),
                                               shuffle=False,
                                               num_workers=0)

    labels = extract_all_labels(training_data=test_config["train_data_path"],
                                validation_data=test_config["validation_data_path"],
                                test_data=test_config["test_data_path"],
                                separator=test_config["data_loader"]["separator"]
                                , label=test_config["data_loader"]["phrase"])

    predictions = predict_simple_composition_model(model_path=argp.model_path, test_data_loader=valid_loader,
                                                   training_config=training_config)
    save_predictions(predictions=predictions, path=prediction_path_dev)
    if test_config["feature_extractor"]["context"] is False:
        feature_extractor = StaticEmbeddingExtractor(
            path_to_embeddings=test_config["feature_extractor"]["static"]["pretrained_model"])
    else:
        bert_parameter = test_config["feature_extractor"]["contextualized"]["bert"]
        bert_model = bert_parameter["model"]
        max_len = bert_parameter["max_sent_len"]
        lower_case = bert_parameter["lower_case"]
        batch_size = bert_parameter["batch_size"]
        feature_extractor = BertExtractor(bert_model=bert_model, max_len=max_len, lower_case=lower_case,
                                          batch_size=batch_size)

    evaluate(embedding_extractor=feature_extractor, predictions=prediction_path_dev, data_loader=valid_loader,
             rank_path=rank_path_dev)

    if test_config["eval_on_test"]:
        prediction_path_test = argp.save_name + "_test_predictions.npy"
        rank_path_test = argp.save_name + "_test_ranks.txt"
        test_loader = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=len(dataset_test),
                                                  shuffle=False,
                                                  num_workers=0)
        predictions = predict_simple_composition_model(model_path=argp.model_path, test_data_loader=test_loader,
                                                       training_config=training_config)
        save_predictions(predictions=predictions, path=prediction_path_test)

        evaluate(embedding_extractor=feature_extractor, predictions=prediction_path_test, data_loader=test_loader,
                 rank_path=rank_path_test)

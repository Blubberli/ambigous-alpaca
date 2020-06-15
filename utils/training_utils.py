import torch
from classification_models import BasicTwoWordClassifier, TransweighTwoWordClassifier, TransferCompClassifier, \
    PhraseContextClassifier, MatrixTwoWordClassifier, MatrixTransferClassifier
from ranking_models import TransweighPretrain, MatrixPretrain, MatrixTransferRanker, TransweighTransferRanker, \
    TransweighJointRanker, MatrixJointRanker, FullAdditive, FullAdditiveJointRanker

from utils import SimplePhraseContextualizedDataset, SimplePhraseStaticDataset, \
    PhraseAndContextDatasetStatic, PhraseAndContextDatasetBert, StaticRankingDataset, ContextualizedRankingDataset

from utils.data_loader import create_label_encoder, extract_all_labels


def init_classifier(config):
    """
    This method initialized the classifier with parameter specified in the config file
    :param config: the configuration
    :return: a torch classifier
    """
    classifier = None
    if config["model"]["type"] == "basic_twoword":
        classifier = BasicTwoWordClassifier(input_dim=config["model"]["input_dim"],
                                            hidden_dim=config["model"]["hidden_size"],
                                            dropout_rate=config["model"]["dropout"],
                                            label_nr=config["model"]["label_size"])
    if config["model"]["type"] == "transweigh_twoword":
        classifier = TransweighTwoWordClassifier(input_dim=config["model"]["input_dim"],
                                                 hidden_dim=config["model"]["hidden_size"],
                                                 dropout_rate=config["model"]["dropout"],
                                                 label_nr=config["model"]["label_size"],
                                                 normalize_embeddings=config["model"]["normalize_embeddings"],
                                                 transformations=config["model"]["transformations"],
                                                 add_single_words=config["model"]["add_single_words"])
    if config["model"]["type"] == "matrix_twoword":
        classifier = MatrixTwoWordClassifier(input_dim=config["model"]["input_dim"],
                                             hidden_dim=config["model"]["hidden_size"],
                                             dropout_rate=config["model"]["dropout"],
                                             label_nr=config["model"]["label_size"],
                                             normalize_embeddings=config["model"]["normalize_embeddings"],
                                             add_single_words=config["model"]["add_single_words"])
    if config["model"]["type"] == "phrase_context":
        classifier = PhraseContextClassifier(embedding_dim=config["model"]["input_dim"],
                                             forward_hidden_dim=config["model"]["hidden_size"],
                                             label_nr=config["model"]["label_size"],
                                             dropout_rate=config["model"]["dropout"],
                                             hidden_size=config["model"]["lstm"]["hidden_size"],
                                             num_layers=config["model"]["lstm"]["layers"])

    if config["model"]["type"] == "transfer_twoword":
        classifier = TransferCompClassifier(input_dim=config["model"]["input_dim"],
                                            hidden_dim=config["model"]["hidden_size"],
                                            dropout_rate=config["model"]["dropout"],
                                            label_nr=config["model"]["label_size"],
                                            normalize_embeddings=config["model"]["normalize_embeddings"],
                                            pretrained_model=config["pretrained_model_path"],
                                            add_single_words=config["model"]["add_single_words"])
    if config["model"]["type"] == "matrix_transfer":
        classifier = MatrixTransferClassifier(input_dim=config["model"]["input_dim"],
                                              hidden_dim=config["model"]["hidden_size"],
                                              dropout_rate=config["model"]["dropout"],
                                              label_nr=config["model"]["label_size"],
                                              normalize_embeddings=config["model"]["normalize_embeddings"],
                                              pretrained_model=config["pretrained_model_path"],
                                              add_single_words=config["model"]["add_single_words"])
    if config["model"]["type"] == "matrix_transfer_ranking":
        classifier = MatrixTransferRanker(dropout_rate=config["model"]["dropout"],
                                          normalize_embeddings=config["model"]["normalize_embeddings"],
                                          pretrained_model=config["pretrained_model_path"])
    if config["model"]["type"] == "transweigh_transfer_ranking":
        classifier = TransweighTransferRanker(dropout_rate=config["model"]["dropout"],
                                              normalize_embeddings=config["model"]["normalize_embeddings"],
                                              pretrained_model=config["pretrained_model_path"])
    if config["model"]["type"] == "transweigh_pretrain":
        classifier = TransweighPretrain(input_dim=config["model"]["input_dim"],
                                        dropout_rate=config["model"]["dropout"],
                                        normalize_embeddings=config["model"]["normalize_embeddings"],
                                        transformations=config["model"]["transformations"])
    if config["model"]["type"] == "matrix_pretrain":
        classifier = MatrixPretrain(input_dim=config["model"]["input_dim"],
                                    dropout_rate=config["model"]["dropout"],
                                    normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "full_additive_pretrain":
        classifier = FullAdditive(input_dim=config["model"]["input_dim"],
                                  dropout_rate=config["model"]["dropout"],
                                  normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "joint_ranking":
        classifier = TransweighJointRanker(input_dim=config["model"]["input_dim"],
                                           dropout_rate=config["model"]["dropout"],
                                           normalize_embeddings=config["model"]["normalize_embeddings"],
                                           transformations=config["model"]["transformations"])
    if config["model"]["type"] == "joint_ranking_matrix":
        classifier = MatrixJointRanker(input_dim=config["model"]["input_dim"],
                                       dropout_rate=config["model"]["dropout"],
                                       normalize_embeddings=config["model"]["normalize_embeddings"])
    if config["model"]["type"] == "joint_ranking_full_additive":
        classifier = FullAdditiveJointRanker(input_dim=config["model"]["input_dim"],
                                       dropout_rate=config["model"]["dropout"],
                                       normalize_embeddings=config["model"]["normalize_embeddings"])

    assert classifier, "no valid classifier name specified in the configuration"
    return classifier


def get_datasets(config):
    """
    Returns the datasets with the corresponding features (defined in the config file)
    :param config: the configuration file
    :return: training, validation, test dataset
    """
    label_encoder = None
    # dataset separator of csv
    separator = config["data_loader"]["separator"]
    label = config["data_loader"]["label"]
    phrase = config["data_loader"]["phrase"]
    # create label encoder if not pretraining:
    if not "pretrain" in config["model"]["type"] and not "ranking" in config["model"]["type"]:
        labels = extract_all_labels(training_data=config["train_data_path"],
                                    validation_data=config["validation_data_path"],
                                    test_data=config["test_data_path"],
                                    separator=separator, label=label)
        label_encoder = create_label_encoder(all_labels=labels)
    # datasets with bert embeddings
    if config["feature_extractor"]["contextualized_embeddings"] is True:
        bert_parameter = config["feature_extractor"]["contextualized"]["bert"]
        bert_model = bert_parameter["model"]
        max_len = bert_parameter["max_sent_len"]
        lower_case = bert_parameter["lower_case"]
        batch_size = bert_parameter["batch_size"]
        context = config["data_loader"]["context"]
        if config["feature_extractor"]["context"] is False:
            if "pretrain" in config["model"]["type"] or "ranking" in config["model"]["type"]:
                mod = config["data_loader"]["modifier"]
                head = config["data_loader"]["head"]
                definition_file = config["data_loader"]["definitions"]
                dataset_train = ContextualizedRankingDataset(data_path=config["train_data_path"],
                                                             bert_model=bert_model, lower_case=lower_case,
                                                             max_len=max_len, separator=separator,
                                                             batch_size=batch_size,
                                                             label=label, mod=mod, head=head,
                                                             label_definition_path=definition_file)
                dataset_valid = ContextualizedRankingDataset(data_path=config["validation_data_path"],
                                                             bert_model=bert_model, lower_case=lower_case,
                                                             max_len=max_len, separator=separator,
                                                             batch_size=batch_size,
                                                             label=label, mod=mod, head=head,
                                                             label_definition_path=definition_file)
                dataset_test = ContextualizedRankingDataset(data_path=config["test_data_path"],
                                                            bert_model=bert_model, lower_case=lower_case,
                                                            max_len=max_len, separator=separator,
                                                            batch_size=batch_size,
                                                            label=label, mod=mod, head=head,
                                                             label_definition_path=definition_file)
            else:
                # phrase only
                dataset_train = SimplePhraseContextualizedDataset(data_path=config["train_data_path"],
                                                                  bert_model=bert_model, lower_case=lower_case,
                                                                  max_len=max_len, separator=separator,
                                                                  batch_size=batch_size, label_encoder=label_encoder,
                                                                  label=label, phrase=phrase, context=context)
                dataset_valid = SimplePhraseContextualizedDataset(data_path=config["validation_data_path"],
                                                                  bert_model=bert_model,
                                                                  lower_case=lower_case, max_len=max_len,
                                                                  separator=separator,
                                                                  batch_size=batch_size, label_encoder=label_encoder,
                                                                  label=label, phrase=phrase, context=context)
                dataset_test = SimplePhraseContextualizedDataset(data_path=config["test_data_path"],
                                                                 bert_model=bert_model,
                                                                 lower_case=lower_case, max_len=max_len,
                                                                 separator=separator,
                                                                 batch_size=batch_size, label_encoder=label_encoder,
                                                                 label=label, phrase=phrase, context=context)
        else:
            # phrase and sentence
            dataset_train = PhraseAndContextDatasetBert(data_path=config["train_data_path"],
                                                        bert_model=bert_model,
                                                        lower_case=lower_case, max_len=max_len,
                                                        batch_size=batch_size, separator=separator,
                                                        tokenizer_model=config["sequence"]["tokenizer"],
                                                        label_encoder=label_encoder,
                                                        label=label, phrase=phrase, context=context)
            dataset_valid = PhraseAndContextDatasetBert(data_path=config["validation_data_path"],
                                                        bert_model=bert_model,
                                                        lower_case=lower_case, max_len=max_len,
                                                        batch_size=batch_size, separator=separator,
                                                        tokenizer_model=config["sequence"]["tokenizer"],
                                                        label_encoder=label_encoder,
                                                        label=label, phrase=phrase, context=context)
            dataset_test = PhraseAndContextDatasetBert(data_path=config["test_data_path"],
                                                       bert_model=bert_model,
                                                       lower_case=lower_case, max_len=max_len,
                                                       batch_size=batch_size, separator=separator,
                                                       tokenizer_model=config["sequence"]["tokenizer"],
                                                       label_encoder=label_encoder,
                                                       label=label, phrase=phrase, context=context)
    # datasets with static embeddings
    else:
        embedding_path = config["feature_extractor"]["static"]["pretrained_model"]
        # phrase only
        if config["feature_extractor"]["context"] is False:
            mod = config["data_loader"]["modifier"]
            head = config["data_loader"]["head"]
            if "pretrain" in config["model"]["type"] or "ranking" in config["model"]["type"]:
                dataset_train = StaticRankingDataset(data_path=config["train_data_path"],
                                                     embedding_path=embedding_path, separator=separator,
                                                     phrase=phrase, mod=mod, head=head)
                dataset_test = StaticRankingDataset(data_path=config["test_data_path"],
                                                    embedding_path=embedding_path, separator=separator,
                                                    phrase=phrase, mod=mod, head=head)
                dataset_valid = StaticRankingDataset(data_path=config["validation_data_path"],
                                                     embedding_path=embedding_path, separator=separator,
                                                     phrase=phrase, mod=mod, head=head)
            else:
                dataset_train = SimplePhraseStaticDataset(data_path=config["train_data_path"],
                                                          embedding_path=embedding_path, separator=separator,
                                                          phrase=phrase, label=label,
                                                          label_encoder=label_encoder)
                dataset_test = SimplePhraseStaticDataset(data_path=config["test_data_path"],
                                                         embedding_path=embedding_path, separator=separator,
                                                         phrase=phrase, label=label,
                                                         label_encoder=label_encoder)
                dataset_valid = SimplePhraseStaticDataset(data_path=config["validation_data_path"],
                                                          embedding_path=embedding_path, separator=separator,
                                                          phrase=phrase, label=label,
                                                          label_encoder=label_encoder)
        else:
            # phrase and sentence
            context = config["data_loader"]["context"]
            dataset_train = PhraseAndContextDatasetStatic(data_path=config["train_data_path"],
                                                          embedding_path=embedding_path, separator=separator,
                                                          phrase=phrase, context=context, label=label,
                                                          tokenizer_model=config["sequence"]["tokenizer"],
                                                          label_encoder=label_encoder)
            dataset_test = PhraseAndContextDatasetStatic(data_path=config["test_data_path"],
                                                         embedding_path=embedding_path, separator=separator,
                                                         phrase=phrase, context=context, label=label,
                                                         tokenizer_model=config["sequence"]["tokenizer"],
                                                         label_encoder=label_encoder)
            dataset_valid = PhraseAndContextDatasetStatic(data_path=config["validation_data_path"],
                                                          embedding_path=embedding_path, separator=separator,
                                                          phrase=phrase, context=context, label=label,
                                                          tokenizer_model=config["sequence"]["tokenizer"],
                                                          label_encoder=label_encoder)

            assert dataset_test and dataset_valid and dataset_train, "there was an error when constructing the " \
                                                                     "datasets"
    return dataset_train, dataset_valid, dataset_test


def convert_logits_to_binary_predictions(logits):
    """
    This method takse raw scores from a binary classifier and converts them into 0 and 1 respectively. The scores
    are expected to be unnormalized, a sigmoid is applied in this method
    :param logits: a list of class scores
    :return: a list of predictions (0 and 1)
    """
    predictions = torch.sigmoid(logits)
    predictions = [0 if x < 0.5 else 1 for x in predictions]
    return predictions

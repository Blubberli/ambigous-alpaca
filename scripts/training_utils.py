from scripts import BasicTwoWordClassifier, TransweighTwoWordClassifier


def init_classifier(config):
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
                                                 transformations=config["model"]["transformations"])

    if not classifier:
        print("no valid classifier name specified in the configuration")
    return classifier

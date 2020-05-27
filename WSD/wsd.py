import pandas as pd
import numpy as np
import random
import argparse
import json
import torch
from collections import defaultdict
from nltk.tokenize import word_tokenize
from utils import StaticEmbeddingExtractor
from utils.training_utils import init_classifier


# @TODO: look at centroid calculation
def description_embedding(description, embeddings, config):
    """
    takes sense description of a lex unit and returns centroid of the words in the description for each lexunit
    :param: description: string of description of a sense
    :param: embeddings: all embeddings
    :param: config
    :return: centroid of vectors of each word in description
    """

    description_tokenised = word_tokenize(description)

    print(description_tokenised)
    description_matrix = np.zeros((len(description_tokenised), config['model']['input_dim']))
    for i, tok in enumerate(description_tokenised):
        description_matrix[i] = embeddings.get_embedding(tok)

    return np.mean(description_matrix, axis=1)  # should be how you calculate centorid but not entirely sure if correct


# @TODO: don't know whether sense_embs as method or directly fed to this method
def adj_triples(wsd_dataset, sense_embs, all_embeddings, all_descriptions):
    """
    takes an adjective and its lexical units and returns a dictionary with the adjective as the key and a triple
    consisting of
    lex_unit_id, sense_embedding, sense_description_embedding
    :param: wsd_dataset: the WSD dataset as a dataframe
    :param: sense embeddings
    """

    adj_senses_and_descrs = dict()
    for _, row in wsd_dataset.iterrows():
        try:
            description = all_descriptions[row['lu']]
            description_emb = description_embedding(description=description, embeddings=all_embeddings, config=config)
        except:
            description_emb = None
        # keys in sense embeddings are encoded as word#lexunit
        vocab_element = row['mod'] + "#" + row['lu']
        sense = sense_embs.get_embedding(vocab_element)
        if row['mod'] not in adj_senses_and_descrs:
            adj_senses_and_descrs[row['mod']] = [(row['lu'], sense, description_emb)]
        else:
            adj_senses_and_descrs[row['mod']].append((row['lu'], sense, description_emb))

    return adj_senses_and_descrs


# @TODO: update to final format
def load_dataset_gerco(path):
    """
    :param: path: loads the Gerco WSD dataset
    :return: df: dataframe of dataset
    """
    df = pd.read_csv(path, delimiter="\t")
    df = df.rename(columns={"ADJ": "mod", "NOUN": "head", "ADJ_LU": "lu", "SENSE_DESCRIPTION": "description",
                            "CONTEXT": "context"})
    return df


def load_dataset_wiki(path):
    """
    Creates a dataframe form the wiktionary wsd dataset
    :param path: path to wiktionary wsd dataset
    :return: a pandas dataframe with modifier, head, lu, description and context
    """
    df = pd.read_csv(path, delimiter="\t")
    adjectives = df["adjektiv"]
    phrases = df["phrase"]
    context = df["context"]
    sense_def = df["sense def"]
    lexical_unit = df["germanet sense"]
    nouns = [p.split(" ")[1] for p in phrases]
    lexical_unit = [unit.split(",")[0].replace("Lexunit(id=", "") for unit in lexical_unit]
    data = {"mod": adjectives, "head": nouns, "lu": lexical_unit, "description": sense_def, "context": context}
    return pd.DataFrame(data)


def read_sense_descriptions(path):
    """
    this method returns a dictionary with a sense and its descriptions
    :param: path: path to sense descirption
    :return: descriptions per sense in a dict
    """
    descriptions = dict()
    with open(path, 'r') as f:
        next(f)
        for l in f:
            l = l.strip()
            line = l.split("\t")
            descriptions[line[1]] = line[2]
    return descriptions


def get_adj2lexunits(path):
    """
    Returns a dictionary of adjectives and corresponding lexical units in Germanet
    :param path: the path to the file that contains the lexical units for adjectives
    :return: a dictionary with adjectives as keys and lexical unit ids as values
    """
    adj2lu = defaultdict(set)
    with open(path, 'r') as f:
        next(f)
        for l in f:
            l = l.strip()
            line = l.split("\t")
            adj2lu[line[0]].add(line[1])
    return adj2lu


# @TODO: not sure what to calculate here...
def get_baseline(wsd_dataset, adj2lexunits):
    """
    this method generates a random baseline for WSD. for each sense we choose a random sense of the adjective
    :param: wsd_dataset: dataframe with WSD dataset
    :return: accuracy
    """
    correct = 0
    false = 0
    for _, row in wsd_dataset.iterrows():
        # get all lus with adjective of current line's adjective
        all_lus = list(adj2lexunits[row['mod']])
        # get correct LU of current line
        correct_lu = row['lu']
        # randomly choose one of all lus for that adjective
        rand_lu = random.choice(all_lus)
        if correct_lu == rand_lu:
            correct += 1
        else:
            false += 1
    accuracy = correct / (correct + false) * 100
    return accuracy


def get_max_lexunit(lexunit_triples, prediction):
    """
    Returns the lexical unit that has the highest similarity to a given vector. the similarity is computed based on
    cosine similarity between the predicted vector and a sense vector, plus the predicted vector and a definition
    embedding if available
    :param lexunit_triples: for a given word, all possible senses with the corresponding representations
    :param prediction: a predicted vector for adj-noun
    :return: the lexical unit with the highest total similarity to the predicted vector
    """
    unit2sim = {}
    for lexunit in lexunit_triples:
        id = lexunit[0]
        sense_emb = lexunit[1]
        def_emb = lexunit[2]
        sense2prediction = np.dot(sense_emb, prediction)
        if def_emb:
            def2prediction = np.dot(def_emb, prediction)
            unit2sim[id] = (sense2prediction + def2prediction) / 2
        else:
            unit2sim[id] = sense2prediction
    similarities = np.array(list(unit2sim.values()))
    return list(unit2sim.keys())[np.argmax(similarities)]


def disambiguate(dataset, triples, predictions):
    """
    This method takes a dataset and predicts the sense for each adjective instance.
    :param dataset: the dataframe that holds a WSD dataset
    :param triples: a dictionary of triples, each triple holding sense vector, definition embedding and id for a
    lexical unit
    :param predictions: the adj-noun vectors composed by a model
    :return: the accuracy for each phrase representation
    """
    final_phrase = predictions["final_phrase_pred"]
    attribute_vec = predictions["attribute"]
    reconstructed_vec = predictions["reconstructed"]
    correct_final_phrase = 0
    correct_attribute = 0
    correct_reconstructed = 0
    for i in range(final_phrase.shape[0]):
        pred_final_phrase = final_phrase[i]
        pred_attribute = attribute_vec[i]
        pred_reconstructed = reconstructed_vec[i]

        adj = dataset["mod"][i]
        correct_lu = dataset["lu"][i]
        possible_senses = triples[adj]
        # triple = lu, sense embeddings, def embedding
        predicted_lu_final_phrase = get_max_lexunit(possible_senses, pred_final_phrase)
        predicted_lu_attribute = get_max_lexunit(possible_senses, pred_attribute)
        predicted_lu_reconstructed = get_max_lexunit(possible_senses, pred_reconstructed)

        if correct_lu == predicted_lu_final_phrase:
            correct_final_phrase += 1
        if correct_lu == predicted_lu_attribute:
            correct_attribute += 1
        if correct_lu == predicted_lu_reconstructed:
            correct_reconstructed += 1
    accuracy_final_phrase = correct_final_phrase / len(dataset) * 100
    accuracy_attribute = correct_attribute / len(dataset) * 100
    accuracy_reconstructed = correct_reconstructed / len(dataset) * 100
    return accuracy_final_phrase, accuracy_attribute, accuracy_reconstructed


def predict_joint_model(model_path, wsd_dataset, training_config, embedding_extractor):
    """
    Predicts the phrases of a given dataset with the joint model
    :param model_path: the path to load the joint model from
    :param wsd_dataset: the dataframe that holds the WSD dataset
    :param training_config: a config object with the parameter the model was trained with
    :param embedding_extractor: the same feature extractor as used in training to extract embeddings for the WSD dataset
    :return: a dictionary with a three vectors for each adj-noun phrase in the WSD dataset, one represents the
    combined representation, one represents the attribute and one the reconstructed phrase
    """
    valid_model = init_classifier(training_config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()
    mod_embeddings = np.array(embedding_extractor.get_array_embeddings(list(wsd_dataset["mod"])))
    head_embeddings = np.array(embedding_extractor.get_array_embeddings(list(wsd_dataset["head"])))

    if valid_model:
        valid_model.load_state_dict(torch.load(model_path))
        valid_model.eval()
        batch = {"w1": torch.from_numpy(mod_embeddings),
                 "w2": torch.from_numpy(head_embeddings), "device": "cpu"}
        composed, rep1, rep2 = valid_model(batch)
        composed = composed.squeeze().to("cpu").detach().numpy()
        rep2 = rep2.squeeze().to("cpu").detach().numpy()
        rep1 = rep1.squeeze().to("cpu").detach().numpy()
        return {"final_phrase_pred": composed, "reconstructed": rep1, "attribute": rep2}


def predict_single_task(model_path, wsd_dataset, training_config, embedding_extractor):
    valid_model = init_classifier(training_config)
    valid_model.load_state_dict(torch.load(model_path))
    valid_model.eval()
    mod_embeddings = np.array(embedding_extractor.get_array_embeddings(list(wsd_dataset["mod"])))
    head_embeddings = np.array(embedding_extractor.get_array_embeddings(list(wsd_dataset["head"])))

    if valid_model:
        valid_model.load_state_dict(torch.load(model_path))
        valid_model.eval()
        batch = {"w1": torch.from_numpy(mod_embeddings),
                 "w2": torch.from_numpy(head_embeddings), "device": "cpu"}
        composed = valid_model(batch)
        composed = composed.squeeze().to("cpu").detach().numpy()

        return {"final_phrase_pred": composed, "reconstructed": composed, "attribute": composed}


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("wsd_dataset", help="the WSD dataset either GerCo or wiktionary")
    argp.add_argument("sense_embeddings", help="retrofitted embeddings for lexical units in GermaNet")
    argp.add_argument("sense_definitions",
                      help="a file that contains sense definitions for lexical units that appear in the WSD dataset")
    argp.add_argument("training_config", help="the config that was used to train the model with")
    argp.add_argument("model_path", help="the path to the model that should be used to construct phrase representations")
    argp = argp.parse_args()

    with open(argp.training_config, 'r') as f:
        training_config = json.load(f)

    adj2lexunits = get_adj2lexunits(argp.sense_definitions)
    descriptions = read_sense_descriptions(argp.sense_definitions)
    if "Gerco" in argp.wsd_dataset:

        wsd_dataset = load_dataset_gerco(argp.wsd_dataset)
    else:
        wsd_dataset = load_dataset_wiki(argp.wsd_dataset)

    embeddings = StaticEmbeddingExtractor(training_config["feature_extractor"]["static"]["pretrained_model"])
    sense_embeddings = StaticEmbeddingExtractor(argp.sense_embeddings)
    triples = adj_triples(wsd_dataset, sense_embeddings, embeddings, descriptions)
    if "pretrain" in training_config["model"]["type"]:
        predictions = predict_single_task(model_path=argp.model_path, training_config=training_config,
                                          wsd_dataset=wsd_dataset, embedding_extractor=embeddings)
    else:
        predictions = predict_joint_model(model_path=argp.model_path, training_config=training_config,
                                          wsd_dataset=wsd_dataset, embedding_extractor=embeddings)

    result_final, result_att, result_reconstructed = disambiguate(wsd_dataset, triples, predictions)
    baseline = get_baseline(wsd_dataset, adj2lexunits)
    print("accuracy for a random picked sense for this dataset is: %.2f" % baseline)
    print(
        "accuracy for final phrase for this dataset is: %.2f\naccuracy for attribute phrase is %.2f \naccuracy for "
        "reconstructed phrase is %.2f" % (result_final, result_att, result_reconstructed))

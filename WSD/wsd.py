import pandas as pd
import numpy as np
import somajo
import random
import argparse
import json
from nltk.tokenize import word_tokenize
from utils import StaticEmbeddingExtractor


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
def adj_triples(wsd_dataset, sense_embs, all_embeddings, config):
    """
    takes an adjective and its lexical units and returns a dictionary with the adjective as the key and a triple consisting of
    lex_unit_id, sense_embedding, sense_description_embedding
    :param: wsd_dataset: the WSD dataset as a dataframe
    :param: sense embeddings
    """

    adj_senses_and_descrs = dict()
    all_descriptions = read_sense_descriptions(config['test_data_path'])
    for _, row in wsd_dataset.iterrows():
        try:
            description = all_descriptions[row['lu']]
            description_emb = description_embedding(description=description, embeddings=all_embeddings, config=config)
        except:
            description_emb = None
        sense = sense_embs[row['lu']]
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


# @TODO: not sure what to calculate here...
def get_baseline(wsd_dataset):
    """
    this method generates a random baseline for WSD. for each sense we choose a random sense of the adjective
    :param: wsd_dataset: dataframe with WSD dataset
    :return: accuracy
    """
    correct = 0
    false = 0
    for _, row in wsd_dataset.iterrows():
        # get all lus with adjective of current line's adjective
        all_lus = wsd_dataset.loc[wsd_dataset['mod'] == row['mod']]
        all_lus = all_lus['lu'].to_list()
        # get correct LU of current line
        correct_lu = row['lu']
        # randomly choose one of all lus for that adjective
        rand_lu = random.choice(all_lus)
        if correct_lu == rand_lu:
            correct += 1
        else:
            false += 1
    accuracy = false / (correct + false) * 100
    return accuracy


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("path_to_config")
    argp = argp.parse_args()

    with open(argp.path_to_config, 'r') as f:  # read in arguments and save them into a configuration object
        config = json.load(f)

    wsd_dataset = load_dataset_gerco(config['train_data_path'])
    baseline = get_baseline(wsd_dataset)
    embeddings = StaticEmbeddingExtractor(config["feature_extractor"]["static"]["pretrained_model"])
    print(config['test_data_path'])
    descriptions = read_sense_descriptions(config['test_data_path'])

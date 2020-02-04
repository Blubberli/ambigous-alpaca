import finalfusion


class StaticEmbeddingExtractor:
    """
    this class includes methods to extract pretrained static word embeddings
    the pretrained embeddings stem from finalfusion
    :param: path_to_embeddings: the path to the pretrained embeddings stored either in .fifu format or .bin (GloVe) or .w2v (word2vec)
    """

    def __init__(self, path_to_embeddings):
        if path_to_embeddings.endswith("fifu"):
            self._embeds = finalfusion.Embeddings(path_to_embeddings, mmap=True)
        elif path_to_embeddings.endswith("bin"):
            self._embeds = finalfusion.Embeddings.read_fasttext(path_to_embeddings, mmap=True)
        elif path_to_embeddings.endswith("w2v"):
            self._embeds = finalfusion.Embeddings.read_fasttext(path_to_embeddings, mmap=True)
        else:
            print("wrong path inserted")

    def get_embedding(self, word):
        """
        takes a word and returns its embedding
        :param word: the word for which an embedding should be returned
        :return: the embedding of the word
        """
        return self._embeds.embedding(word)

    def get_array_embeddings(self, array_words):
        """
        takes an array of words and returns an array of embeddings of those words
        :param array_words: a word array of length x
        :return: array_embeddings: the embeddings of those words in an array of length x
        """
        array_embeddings = []
        [array_embeddings.append(self._embeds.embedding(words)) for words in array_words]
        return array_embeddings

    @property
    def embeds(self):
        return self._embeds

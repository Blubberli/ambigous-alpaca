import finalfusion


class StaticEmbeddingExtractor:
    """
    this class includes methods to extract pretrained static word embeddings
    the pretrained embeddings stem from finalfusion
    :param: path_to_fifu: the path to the pretrained embeddings stored with .fifu format
    """

    def __init__(self, path_to_fifu):
        self._embeds = finalfusion.Embeddings(path_to_fifu, mmap=True)

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

import finalfusion


class StaticEmbeddingExtractor:

    def __init__(self, path_to_fifu):
        self._embeds = finalfusion.Embedding(path_to_fifu, mmap=True)

    def get_embedding(self, word):
        return self._embeds.embedding(word)


    @property
    def embeds(self):
        return self._embeds

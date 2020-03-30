from scripts.composition_functions import transweigh, transweigh_weight, tw_transform, concat
from scripts.basic_twoword_classifier import BasicTwoWordClassifier
from scripts.feature_extractor_contextualized import BertExtractor
from scripts.loss_functions import multi_class_cross_entropy, binary_class_cross_entropy, get_loss_cosine_distance

from scripts.logger_config import create_config

from scripts.feature_extractor_static import StaticEmbeddingExtractor
from scripts.transweigh_twoword_classifier import TransweighTwoWordClassifier
from scripts.transfer_comp_classifier import TransferCompClassifier
from scripts.data_loader import SimplePhraseStaticDataset, SimplePhraseContextualizedDataset, \
    PhraseAndContextDatasetStatic, PhraseAndContextDatasetBert


from scripts.data_loader import SimplePhraseContextualizedDataset, SimplePhraseStaticDataset
from scripts.phrase_context_classifier import PhraseContextClassifier
from scripts.transweigh_pretrain import TransweighPretrain
from scripts.matrix_pretrain import MatrixPretrain
from scripts.matrix_twoword_classifier import MatrixTwoWordClassifier
from scripts.matrix_transfer_classifier import MatrixTransferClassifier

from scripts.ranking import Ranker
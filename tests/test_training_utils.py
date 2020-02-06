import unittest
import json
import pathlib
import numpy as np
from scripts import BasicTwoWordClassifier, TransweighTwoWordClassifier, training_utils


class TrainingUtilsTest(unittest.TestCase):
    """
    this class tests the training utils script
    This test suite can be ran with:
        python -m unittest -q tests.TrainingUtilsTest
    """

    def setUp(self):
        config_path = str(pathlib.Path(__file__).parent.absolute().joinpath("test_config.json"))
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def test_classifier_init(self):
        """tests whether the correct classifier can be initialized and has the correct classifier type and learning
        rate"""
        classifier_basic = BasicTwoWordClassifier(hidden_dim=5, input_dim=2, dropout_rate=0.1, label_nr=3)
        classifier_tw = TransweighTwoWordClassifier(hidden_dim=5, input_dim=2, dropout_rate=0.1,
                                                    normalize_embeddings=True, label_nr=3, transformations=2)
        classifier = training_utils.init_classifier(self.config)
        np.testing.assert_equal(type(classifier_basic), type(classifier))
        np.testing.assert_equal(type(classifier_tw) == type(classifier), False)
        np.testing.assert_equal(classifier.dropout_rate, 0.2)
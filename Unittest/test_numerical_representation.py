import unittest
import numpy as np
from Src.numerical_representation import EmbeddingGenerator

class TestEmbeddingGenerator(unittest.TestCase):
    """
    This class contains unit tests for the EmbeddingGenerator class.
    """

    def setUp(self):
        """
        Set up the EmbeddingGenerator instance.
        """
        self.embedding_generator = EmbeddingGenerator()

    def test_get_esmv1_embedding(self):
        sequence = 'ACDEFGHIKLMNPQRSTVWY'
        embedding = self.embedding_generator.get_esmv1_embedding(sequence)
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)

    def test_get_prott5_embedding(self):
        sequence = 'ACDEFGHIKLMNPQRSTVWY'
        embedding = self.embedding_generator.get_prott5_embedding(sequence)
        self.assertIsNotNone(embedding)
        self.assertIsInstance(embedding, np.ndarray)

if __name__ == '__main__':
    unittest.main()
from ifeatpro.features import get_all_features
from aaindex import aaindex1
from transformers import T5Tokenizer, T5EncoderModel
import torch
import esm 
import numpy as np

class SequenceRepresentation:
    """
    Represents a sequence and provides methods for numerical representation.
    """

    def __init__(self, sequence):
        """
        Initializes a SequenceRepresentation object.

        Parameters:
        sequence (str): The sequence to be represented numerically.
        """
        self.sequence = sequence

    def one_hot_encoding(self):
        """
        Performs one-hot encoding on the sequence.

        Returns:
        numpy.ndarray: The one-hot encoded matrix representation of the sequence.
        """
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_index = dict((c, i) for i, c in enumerate(amino_acids))

        one_hot_matrix = np.zeros((len(self.sequence), len(amino_acids)))
        for i, aa in enumerate(self.sequence):
            one_hot_matrix[i, aa_to_index[aa]] = 1
        
        return one_hot_matrix
    
    def get_ifeatpro_features(self):
        """
        Retrieves the IFeatPro features for the sequence.

        Returns:
        list: The IFeatPro features for each amino acid in the sequence.
        """
        return get_all_features(self)
    
    def get_aaindex(self):
        """
        Retrieves the AAIindex values for the sequence.

        Returns:
        list: The AAIindex values for each amino acid in the sequence.
        """
        aaindex_values = []
        for aa in self.sequence:
            aaindex_values.append(aaindex1[aa])
        return aaindex_values

class EmbeddingGenerator:
    """
    Class for generating embeddings using ESM-1V and ProtT5-XL-UniRef50 models.
    """

    def __init__(self, device='cuda'):
        """
        Initializes the EmbeddingGenerator.

        Args:
            device (str): Device to use for computation. Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load ESM-1V model
        self.esm_model, self.esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.esm_model = self.esm_model.to(self.device)
        self.batch_converter = self.esm_alphabet.get_batch_converter()

        # Load ProtT5-XL-UniRef50 model
        self.t5_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.t5_model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.t5_model = self.t5_model.to(self.device)

    def get_esmv1_embedding(self, sequence):
        """
        Generates ESM-1V embedding for a given sequence.

        Args:
            sequence (str): Input sequence.

        Returns:
            numpy.ndarray: ESM-1V embedding of the sequence.
        """
        data = [(f'seq1', sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(batch_tokens.to(self.device), repr_layers=[33], return_contacts=False)
            token_representations = results['representations'][33]
            sequence_embedding = token_representations.mean(1).cpu().numpy()
        return sequence_embedding
    
    def get_prott5_embedding(self, sequence):
        """
        Generates ProtT5-XL-UniRef50 embedding for a given sequence.

        Args:
            sequence (str): Input sequence.

        Returns:
            numpy.ndarray: ProtT5-XL-UniRef50 embedding of the sequence.
        """
        inputs = self.t5_tokenizer(sequence, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.t5_model(**inputs)

        sequence_embedding = outputs.last_hidden_state.mean(1).cpu().numpy()
        return sequence_embedding

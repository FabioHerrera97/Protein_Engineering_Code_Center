from ifeatpro.features import get_all_features
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.preprocessing import OneHotEncoder
import torch
import esm 

class OneHotEncoding:

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, categories=[list('ACDEFGHIKLMNPQRSTVWY')])
    
    def encode(self, sequence):
        sequence = [[aa] for aa in sequence]
        return self.encoder.fit_transform(sequence).flatten()
    
class IfeatproEncoding:

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def get_fasta_file(self, sequences):
        with open(f'{self.output_dir}/sequences.fasta', 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f'>seq{i}\n{seq}\n')

    def get_ifeatpro_features(self):
        get_all_features(f'{self.output_dir}/sequences.fasta', self.output_dir)

class Esm1v_Encoding:

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.esm_model, self.esm_alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
        self.esm_model = self.esm_model.to(self.device)
        self.batch_converter = self.esm_alphabet.get_batch_converter()

    def calculate_esmv1_embedding(self, sequence):
        data = [(f'seq1', sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(batch_tokens.to(self.device), repr_layers=[33], return_contacts=False)
            token_representations = results['representations'][33]
            sequence_embedding = token_representations.mean(1).cpu().numpy()
        return sequence_embedding
    
    def generate_esm1v_embedding(self, sequence):
        esm1v_embedding = self.calculate_esmv1_embedding(sequence)
        return esm1v_embedding


class Prott5Encoding:
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.t5_tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.t5_model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.t5_model = self.t5_model.to(self.device)
    
    def calculate_prott5_embedding(self, sequence):
        inputs = self.t5_tokenizer(sequence, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.t5_model(**inputs)

        sequence_embedding = outputs.last_hidden_state.mean(1).cpu().numpy()
        return sequence_embedding
    
    def generate_prott5_embedding(self, sequence):
        prott5_embedding = self.calculate_prott5_embedding(sequence)
        return prott5_embedding
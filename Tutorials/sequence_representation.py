import pandas as pd
import numpy as np
import argparse
import h5py
import sys 

sys.path.insert(1, '/raid/data/fherrera/Protein_Engineering_Code_Center/Src/')
from numerical_representation import SequenceRepresentation, EmbeddingGenerator


def get_sequences(data_file, seq_column, id_column, label_column):

    data = pd.read_csv(data_file)
    sequences = data[seq_column]
    label = np.array(label[0] for label in data[label_column])
    ids = np.array(id[0] for id in data[id_column])

    return sequences, label, ids

def get_representations(sequences, label, ids, feature_types, output_file):
    """
    Generate representations for the given sequences based on the specified feature types.

    Args:
        sequences (list): List of input sequences.
        label (str): Label for the representations.
        ids (list): List of IDs corresponding to the sequences.
        feature_types (list): List of feature types to generate representations for.
            Valid options: 'one_hot', 'ifeatpro', 'aaindex', 'esmv1', 'prott5', 'all'.
        output_file (str): Output file path to save the generated representations.

    Raises:
        ValueError: If an invalid feature type is provided.

    Returns:
        None
    """

    if 'one_hot' or 'all' in feature_types:
        one_hot_features = [SequenceRepresentation.one_hot_encoding(sequence) for sequence in sequences]
        h5_file = f'{output_file}_one_hot.h5'
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('ids', data=ids)
            f.create_dataset('labels', data=label)
            f.create_dataset('features', data=one_hot_features)

    if 'ifeatpro' or 'all' in feature_types:
        ifeatpro_features = [SequenceRepresentation.get_ifeatpro_features(sequence) for sequence in sequences]
        h5_file = f'{output_file}_ifeatpro.h5'
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('ids', data=ids)
            f.create_dataset('labels', data=label)
            f.create_dataset('features', data=ifeatpro_features)    

    if 'aaindex' or 'all' in feature_types:
        aaindex_features = [SequenceRepresentation.get_aaindex(sequence) for sequence in sequences]
        h5_file = f'{output_file}_aaindex.h5'
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('ids', data=ids)
            f.create_dataset('labels', data=label)
            f.create_dataset('features', data=aaindex_features)

    if 'esmv1' or 'all' in feature_types:
        esmv1_features = [EmbeddingGenerator.get_esmv1_embedding(sequence) for sequence in sequences]
        h5_file = f'{output_file}_esmv1.h5'
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('ids', data=ids)
            f.create_dataset('labels', data=label)
            f.create_dataset('features', data=esmv1_features)   

    if 'prott5' or 'all' in feature_types:
        prot5_features = [EmbeddingGenerator.get_prott5_embedding(sequence) for sequence in sequences]
        h5_file = f'{output_file}_prott5.h5'
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('ids', data=ids)
            f.create_dataset('labels', data=label)
            f.create_dataset('features', data=prot5_features)   

    else:
        raise ValueError('Invalid feature type. Please choose from one_hot, ifeatpro, aaindex, esmv1, prott5, or all.')


def main():
    parser = argparse.ArgumentParser(description='Generate numerical representations for protein sequences.')
    parser.add_argument('--data_file', type=str, help='Path to the data file containing sequences.')
    parser.add_argument('--seq_column', type=str, help='Name of the column containing sequences.')
    parser.add_argument('--id_column', type=str, help='Name of the column containing sequence IDs.')
    parser.add_argument('--label_column', type=str, help='Name of the column containing labels.')
    parser.add_argument('--feature_types', type=str, nargs='+', help='Types of features to generate (one_hot, ifeatpro, aaindex, esmv1, prott5, all).')
    parser.add_argument('--output_file', type=str, help='Path to the output file to save the features.')
    args = parser.parse_args()

    sequences, label, ids = get_sequences(args.data_file, args.seq_column, args.id_column, args.label_column)
    get_representations(sequences, label, ids, args.feature_types, args.output_file)

if __name__ == '__main__':
    main()

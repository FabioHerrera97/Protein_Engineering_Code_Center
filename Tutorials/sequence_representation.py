import pandas as pd
import argparse
from Src.numerical_representation import SequenceRepresentation, EmbeddingGenerator
import h5py

def get_sequences(data_file, seq_column, id_column, label_column):

    data = pd.read_csv(data_file)
    sequences = data[seq_column]
    label = data[label_column]
    ids = data[id_column]

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

    feature_generator = SequenceRepresentation()
    embedding_generator = EmbeddingGenerator()

    if 'one_hot' or 'all' in feature_types:
        one_hot_features = [feature_generator.one_hot_encoding(sequence) for sequence in sequences]
        save_features_to_h5(ids, label, one_hot_features, output_file, 'one_hot')

    if 'ifeatpro' or 'all' in feature_types:
        ifeatpro_features = [feature_generator.get_ifeatpro_features(sequence) for sequence in sequences]
        save_features_to_h5(ids, label, ifeatpro_features, output_file, 'ifeatpro')

    if 'aaindex' or 'all' in feature_types:
        aaindex_features = [feature_generator.get_aaindex(sequence) for sequence in sequences]
        save_features_to_h5(ids, label, aaindex_features, output_file, 'aaindex')

    if 'esmv1' or 'all' in feature_types:
        esmv1_features = [embedding_generator.get_esmv1_embedding(sequence) for sequence in sequences]
        save_features_to_h5(ids, label, esmv1_features, output_file, 'esmv1')

    if 'prott5' or 'all' in feature_types:
        prot5_features = [embedding_generator.get_prott5_embedding(sequence) for sequence in sequences]
        save_features_to_h5(ids, label, prot5_features, output_file, 'prott5')

    else:
        raise ValueError('Invalid feature type. Please choose from one_hot, ifeatpro, aaindex, esmv1, prott5, or all.')

def save_features_to_h5(ids, label, features, output_file, feature_type):
    with h5py.File(output_file, 'a') as f:
        f.create_dataset(f'{feature_type}/ids', data=ids)
        f.create_dataset(f'{feature_type}/labels', data=label)
        f.create_dataset(f'{feature_type}/features', data=features)

def main():
    parser = argparse.ArgumentParser(description='Generate numerical representations for protein sequences.')
    parser.add_argument('data_file', type=str, help='Path to the data file containing sequences.')
    parser.add_argument('seq_column', type=str, help='Name of the column containing sequences.')
    parser.add_argument('id_column', type=str, help='Name of the column containing sequence IDs.')
    parser.add_argument('label_column', type=str, help='Name of the column containing labels.')
    parser.add_argument('feature_types', type=str, nargs='+', help='Types of features to generate (one_hot, ifeatpro, aaindex, esmv1, prott5, all).')
    parser.add_argument('output_file', type=str, help='Path to the output file to save the features.')
    args = parser.parse_args()

    sequences, label, ids = get_sequences(args.data_file, args.seq_column, args.id_column, args.label_column)
    get_representations(sequences, label, ids, args.feature_types, args.output_file)

import pandas as pd
import numpy as np
import argparse
import sys 
import h5py

sys.path.insert(1, '/raid/data/fherrera/Protein_Engineering_Code_Center/Src/')
from numerical_representation import OneHotEncoding, IfeatproEncoding, Esm1v_Encoding, Prott5Encoding


def get_representations(data_file, seq_column, feature_types, output_dir):
    """
    Generate representations for the given sequences based on the specified feature types.

    Args:
        sequences (list): List of input sequences.
        feature_types (list): List of feature types to generate representations for.
            Valid options: 'one_hot', 'ifeatpro', 'esmv1', 'prott5', 'all'.
        output_file (str): Output file path to save the generated representations.

    Raises:
        ValueError: If an invalid feature type is provided.

    Returns:
        None

    """

    data = pd.read_csv(data_file)
    sequences = data[seq_column]

    if 'one_hot' in feature_types or 'all' in feature_types:
        one_hot_encoder = OneHotEncoding()
        one_hot_representations = [one_hot_encoder.encode(sequence) for sequence in sequences]
        df_one_hot = pd.DataFrame(one_hot_representations)
        df_one_hot.to_csv(f'{output_dir}/one_hot.csv', index=False)
        print('One hot encoding done!')

    if 'ifeatpro' in feature_types or 'all' in feature_types:
        ifeatpro_encoder = IfeatproEncoding(output_dir)
        ifeatpro_encoder.get_fasta_file(sequences)
        ifeatpro_encoder.get_ifeatpro_features()
        print('Ifeatpro encoding done!')

    if 'esm1v' in feature_types or 'all' in feature_types:
        esm1v_encoder = Esm1v_Encoding()
        esmv1_features = [esm1v_encoder.generate_esm1v_embedding(sequence) for sequence in sequences]
        esmv1_features = np.array(esmv1_features)
        with h5py.File(f'{output_dir}/esm1v.h5', 'w') as f:
            f.create_dataset('esm1v', data=esmv1_features)
        print('Esmv1 encoding done!')

    if 'prott5' in feature_types or 'all' in feature_types:
        prott5_encoder = Prott5Encoding()
        prott5_features = [prott5_encoder.generate_prott5_embedding(sequence) for sequence in sequences]
        prott5_features = np.array(prott5_features)
        with h5py.File(f'{output_dir}/prott5.h5', 'w') as f:
            f.create_dataset('prott5', data=prott5_features)
        print('Prott5 encoding done!')

    if not any(feature_type in ['one_hot', 'ifeatpro', 'esm1v', 'prott5', 'all'] for feature_type in feature_types):
        raise ValueError('Invalid feature type. Please choose from one_hot, ifeatpro, esm1v, prott5, or all.')

def main():
    parser = argparse.ArgumentParser(description='Generate numerical representations for protein sequences.')
    parser.add_argument('--data_file', type=str, help='Path to the data file containing sequences.')
    parser.add_argument('--seq_column', type=str, help='Name of the column containing sequences.')
    parser.add_argument('--feature_types', type=str, nargs='+', help='Types of features to generate (one_hot, ifeatpro, esm1v, prott5, all).')
    parser.add_argument('--output_dir', type=str, help='Path to the output file to save the features.')
    args = parser.parse_args()

    get_representations(args.data_file, args.seq_column, args.feature_types, args.output_dir)

if __name__ == '__main__':
    main()

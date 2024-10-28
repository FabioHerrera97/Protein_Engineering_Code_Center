import pandas as pd
import numpy as np
import argparse
import sys 

sys.path.insert(1, '/raid/data/fherrera/Protein_Engineering_Code_Center/Src/')
from numerical_representation import OneHotEncoding, SequenceRepresentation, EmbeddingGenerator



def get_representations(data_file, seq_column, feature_types, output_dir):
    """
    Generate representations for the given sequences based on the specified feature types.

    Args:
        sequences (list): List of input sequences.
        feature_types (list): List of feature types to generate representations for.
            Valid options: 'one_hot', 'ifeatpro', 'aaindex', 'esmv1', 'prott5', 'all'.
        output_file (str): Output file path to save the generated representations.

    Raises:
        ValueError: If an invalid feature type is provided.

    Returns:
        None

    """

    data = pd.read_csv(data_file)
    sequences = data[seq_column]

    if 'one_hot' in feature_types:

        one_hot_encoder = OneHotEncoding()
        one_hot_representations = [one_hot_encoder.encode(sequence) for sequence in sequences]
        df_one_hot = pd.DataFrame(one_hot_representations)
        df_one_hot.to_csv(f'{output_dir}/one_hot.csv', index=False)


    '''

    if 'ifeatpro' or 'all' in feature_types:
        ifeatpro_features = [SequenceRepresentation.get_ifeatpro_features(sequence, output_file) for sequence in sequences]
        ifeatpro_features_df = pd.DataFrame(ifeatpro_features)
        out_file = f'{output_file}_ifeatpro.pkl'
        ifeatpro_features_df.loc[:, 'Ids', 'Log_expression'] = ids, label  
        ifeatpro_features_df.to_pickle(out_file, index=False) 
          

    if 'aaindex' or 'all' in feature_types:
        aaindex_features = [SequenceRepresentation.get_aaindex(sequence) for sequence in sequences]
        aaindex_features_df = pd.DataFrame(aaindex_features)
        out_file = f'{output_file}_aaindex.pkl'
        aaindex_features_df.loc[:, 'Ids', 'Log_expression'] = ids, label   
        aaindex_features_df.to_pickle(out_file, index=False)  

    if 'esmv1' or 'all' in feature_types:
        esmv1_features = [EmbeddingGenerator.get_esmv1_embedding(sequence) for sequence in sequences]
        esmv1_features_df = pd.DataFrame(esmv1_features)
        out_file = f'{output_file}_esmv1.pkl'
        esmv1_features_df.loc[:, 'Ids', 'Log_expression'] = ids, label
        esmv1_features_df.to_pickle(out_file, index=False)

    if 'prott5' or 'all' in feature_types:
        prot5_features = [EmbeddingGenerator.get_prott5_embedding(sequence) for sequence in sequences]
        prot5_features_df = pd.DataFrame(prot5_features)
        out_file = f'{output_file}_prott5.pkl'
        prot5_features_df.loc[:, 'Ids', 'Log_expression'] = ids, label 
        prot5_features_df.to_pickle(out_file, index=False)

    else:
        raise ValueError('Invalid feature type. Please choose from one_hot, ifeatpro, aaindex, esmv1, prott5, or all.')'''


def main():
    parser = argparse.ArgumentParser(description='Generate numerical representations for protein sequences.')
    parser.add_argument('--data_file', type=str, help='Path to the data file containing sequences.')
    parser.add_argument('--seq_column', type=str, help='Name of the column containing sequences.')
    parser.add_argument('--feature_types', type=str, nargs='+', help='Types of features to generate (one_hot, ifeatpro, aaindex, esmv1, prott5, all).')
    parser.add_argument('--output_dir', type=str, help='Path to the output file to save the features.')
    args = parser.parse_args()

    get_representations(args.data_file, args.seq_column, args.feature_types, args.output_dir)

if __name__ == '__main__':
    main()

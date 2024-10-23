# Generating protein sequences numerical representations

To generate the sequence numerical representation run the code below: 

```
python sequence_representation.py --data_file ../Data/raw_data/Alpha_Amylase.csv --seq_column mutated_sequence --id_column Mutation --label_column Expression --feature_types all --output_file encoded_dataset.h5 
```

These are the arguments that must be supplied to run the script:

`--data_file`: Input file containing the experimental data

`--seq_column`: Column containing the mutant sequences

`--id_column`: Column containing the mutant code

`--label_column`: Column containing the experimental labels

`--feature_types`: Types of features to generate (one_hot, ifeatpro, aaindex, esmv1, prott5, all)

This script produces as output a `.h5` with the representations stored at `../data`. 

If more representations need to be created apart from the 5 representations presented here, modify the script `sequence_representation.py`

References

1. Zhen et al. iFeature: a Python package and web server for features extraction and selection from protein and peptide sequences, Bioinformatics, Volume 34, Issue 14, July 2018, Pages 2499–2502, https://doi.org/10.1093/bioinformatics/bty140
2. Shuichi Kawashima, Minoru Kanehisa, AAindex: Amino Acid index database, Nucleic Acids Research, Volume 28, Issue 1, 1 January 2000, Page 374, https://doi.org/10.1093/nar/28.1.374
3. Meier et al. Language models enable zero-shot prediction of the effects of mutations on protein function, Advances in Neural Information Processing Systems, Volume 34, 2021, https://proceedings.neurips.cc/paper_files/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf
4. A. Elnaggar et al., ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning, in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7112-7127, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3095381.

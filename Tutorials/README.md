# Generating protein sequences numerical representations

To generate the sequence numerical representation run the code below: 

```
python sequence_representation.py --data_file ../Data/Cleaned_data/Alpha_Amylase_cleaned.csv --seq_column mutated_sequence --feature_types all --output_dir ../Data/numeric_representations 
```

These are the arguments that must be supplied to run the script:

`--data_file`: Input file containing the experimental data

`--seq_column`: Column containing the mutant sequences

`--feature_types`: Types of features to generate (one_hot, ifeatpro, aaindex, esmv1, prott5, all)

`--output_dir`: Path to store the files with the numerical representations

This script produces as output a `.h5` with the representations stored at `../data`. 

If more representations need to be created apart from the 4 representations presented here, they can be added as classes in the script `sequence_representation.py`

## References

1. Zhen et al. iFeature: a Python package and web server for features extraction and selection from protein and peptide sequences, Bioinformatics, Volume 34, Issue 14, July 2018, Pages 2499–2502, https://doi.org/10.1093/bioinformatics/bty140
2. Meier et al. Language models enable zero-shot prediction of the effects of mutations on protein function, Advances in Neural Information Processing Systems, Volume 34, 2021, https://proceedings.neurips.cc/paper_files/paper/2021/file/f51338d736f95dd42427296047067694-Paper.pdf
3. A. Elnaggar et al., ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning, in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7112-7127, 1 Oct. 2022, doi: 10.1109/TPAMI.2021.3095381.

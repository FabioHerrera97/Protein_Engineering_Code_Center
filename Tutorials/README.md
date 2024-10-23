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

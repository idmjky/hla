import pandas as pd
import numpy as np

# Read the CSV files
scores_df = pd.read_csv('L471_score.csv')
hla_freq_df = pd.read_csv('HLA_Frequency.csv')

# Get the HLA allele columns (they start with A*, B*, or C*)
hla_columns = [col for col in scores_df.columns if col.startswith(('A*', 'B*', 'C*'))]
A_columns = [col for col in hla_columns if col.startswith('A*')]
B_columns = [col for col in hla_columns if col.startswith('B*')]
C_columns = [col for col in hla_columns if col.startswith('C*')]

# Apply ReLU function to scores (replace values below 3 with 0)
for col in hla_columns:
    scores_df[col] = scores_df[col].apply(lambda x: max(0, x - 3))

# Normalize scores for each HLA allele using min-max normalization
for col in hla_columns:
    max_score = scores_df[col].max()
    min_score = scores_df[col].min()
    if max_score > min_score:  # Avoid division by zero
        scores_df[col] = (scores_df[col] - min_score) / (max_score - min_score)

# Create a dictionary of HLA frequencies
hla_freq_dict = dict(zip(hla_freq_df['Allele'], hla_freq_df['percentage']))

# Normalize HLA frequencies for each subtype
A_freq_dict = {k: v for k, v in hla_freq_dict.items() if k.startswith('A*')}
B_freq_dict = {k: v for k, v in hla_freq_dict.items() if k.startswith('B*')}
C_freq_dict = {k: v for k, v in hla_freq_dict.items() if k.startswith('C*')}

total_A = sum(A_freq_dict.values())
total_B = sum(B_freq_dict.values())
total_C = sum(C_freq_dict.values())

A_freq_dict = {k: v/total_A for k, v in A_freq_dict.items()}
B_freq_dict = {k: v/total_B for k, v in B_freq_dict.items()}
C_freq_dict = {k: v/total_C for k, v in C_freq_dict.items()}

# Calculate weighted scores for each peptide and each subtype
def calc_weighted_score(row, columns, freq_dict):
    score = 0
    for hla in columns:
        if hla in freq_dict:
            score += row[hla] * freq_dict[hla]
    return score

scores_df['weighted_score_A'] = scores_df.apply(lambda row: calc_weighted_score(row, A_columns, A_freq_dict), axis=1)
scores_df['weighted_score_B'] = scores_df.apply(lambda row: calc_weighted_score(row, B_columns, B_freq_dict), axis=1)
scores_df['weighted_score_C'] = scores_df.apply(lambda row: calc_weighted_score(row, C_columns, C_freq_dict), axis=1)

# Select and reorder columns
output_columns = ['name2', 'AA', 'gene', 'mutation', 'mpos', 'spos', 'kind', 'wt_AA', 'mut_AA',
                 'weighted_score_A', 'weighted_score_B', 'weighted_score_C']
output_df = scores_df[output_columns]

# Save to new CSV file
output_df.to_csv('L471_weighted_score.csv', index=False)

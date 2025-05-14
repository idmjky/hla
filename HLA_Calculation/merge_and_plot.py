import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text

# Read the CSV files
wt_df = pd.read_csv('L471_weighted_wt_score.csv')
mut_df = pd.read_csv('L471_weighted_score.csv')

# Define HLA subtypes
subtypes = ['A', 'B', 'C']

# Merge the dataframes on name2
merged_df = pd.merge(wt_df, mut_df, on='name2', suffixes=('_wt', '_mut'))

# Epsilon to avoid log(0)
epsilon = 1e-10

# For each subtype, calculate ratio, log-transform, plot
for subtype in subtypes:
    wt_col = f'weighted_score_{subtype}_wt'
    mut_col = f'weighted_score_{subtype}_mut'
    # Replace zeros with epsilon
    merged_df[wt_col] = merged_df[wt_col].replace(0, epsilon)
    merged_df[mut_col] = merged_df[mut_col].replace(0, epsilon)
    # Calculate ratio
    ratio_col = f'score_ratio_{subtype}'
    merged_df[ratio_col] = merged_df[mut_col] / merged_df[wt_col]
    # Log-transform
    log_wt_col = f'log_{wt_col}'
    log_mut_col = f'log_{mut_col}'
    merged_df[log_wt_col] = np.log10(merged_df[wt_col])
    merged_df[log_mut_col] = np.log10(merged_df[mut_col])
    # Top 10 by ratio
    top_10 = merged_df.nlargest(10, ratio_col)
    # Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=merged_df, x=log_wt_col, y=log_mut_col, alpha=0.5)
    plt.scatter(top_10[log_wt_col], top_10[log_mut_col], color='red', s=100, label='Top 10 by ratio')
    # Add labels for top 10
    texts = []
    for idx, row in top_10.iterrows():
        label = f"{row.get('gene_mut', row.get('gene', ''))}_{row.get('mutation_mut', row.get('mutation', ''))}"
        texts.append(
            plt.text(row[log_wt_col], row[log_mut_col], label, fontsize=10, weight='bold', color='black')
        )
    adjust_text(
        texts,
        arrowprops=None,
        expand_points=(2, 2),
        expand_text=(1.2, 1.2),
        force_text=0.5,
        force_points=0.5
    )
    plt.xlabel(f'log10(Wild-type Weighted Score)')
    plt.ylabel(f'log10(Mutant Weighted Score)')
    plt.title(f'Mutant vs Wild-type Weighted Scores - HLA- {subtype}')
    plt.savefig(f'score_comparison_{subtype}.png', dpi=300, bbox_inches='tight')
    plt.close()
    # Print top 10 for this subtype
    print(f"\nTop 10 entries by score ratio for HLA {subtype}:")
    print(top_10[['name2', wt_col, mut_col, ratio_col]].to_string())

# Save the merged data with all ratios
merged_df.to_csv('merged_scores.csv', index=False) 
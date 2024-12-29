import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Mention the Dataset name
x = 'GSE18842.xlsx'

# Load the Dataset using pandas
df = pd.read_excel(x)
# print(df.head(5))

# Define the keys to specify 'Healthy' and 'Disease' samples
Healthy = 'Control'
Disease = 'Tumor'

# Preprocessing
df = df.drop(['!Sample_title'], axis=1)
df = df.dropna(subset=['Gene Symbol'])

# Handle the Duplicate rows based on Gene name
duplicates = df[df.duplicated(subset='Gene Symbol', keep=False)]
avg_duplicates = duplicates.groupby('Gene Symbol').mean().reset_index()
filtered_df = df[~df.duplicated(subset='Gene Symbol', keep=False)]
df = pd.concat([filtered_df, avg_duplicates])

# Normalize the matrix
scaler = MinMaxScaler()
normalized = scaler.fit_transform(df.iloc[:, 1:])
normalized_df = pd.DataFrame(normalized, columns=df.columns[1:]).reset_index(drop=True)
normalized_df['Gene Symbol'] = df['Gene Symbol'].reset_index(drop=True)
normalized_df.set_index('Gene Symbol', inplace=True)

# Transpose the matrix
transposed_df = normalized_df.transpose()

# Get sample labels and conditions
sample_labels = transposed_df.index
conditions = ['Tumor' if 'Tumor' in label else 'Control' for label in sample_labels]

# Separate data into Tumor and control groups
hcm_samples = transposed_df.loc[[label for label, condition in zip(sample_labels, conditions) if condition == 'Tumor']]
control_samples = transposed_df.loc[[label for label, condition in zip(sample_labels, conditions) if condition == 'Control']]

# Perform t-test for each gene
p_values = []
logFC_values = []

for gene in transposed_df.columns:
    hcm_values = hcm_samples[gene]
    control_values = control_samples[gene]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(hcm_values, control_values, equal_var=False)
    
    # Calculate log fold change
    logFC = np.log2(hcm_values.mean() / control_values.mean())
    
    p_values.append(p_value)
    logFC_values.append(logFC)

# Adjust p-values for multiple testing
_, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')

# Create a results DataFrame
results_df = pd.DataFrame({
    'logFC': logFC_values,
    'P.Value': p_values,
    'adj.P.Val': adjusted_p_values
}, index=transposed_df.columns)

# Print the results
print(results_df.head(5))
# results_df.to_excel("Pval_and_log2FC.xlsx")

# Adding columns for significant genes based on thresholds
logFC_threshold = 1
p_value_threshold = 0.05

results_df['-log10(P.Value)'] = -np.log10(results_df['P.Value'])
results_df['significant'] = (abs(results_df['logFC']) > logFC_threshold) & (results_df['adj.P.Val'] < p_value_threshold)
results_df['regulation'] = np.where(results_df['logFC'] > 0, 'Up', 'Down')

# Extract significant genes for up-regulation and down-regulation
significant_up_genes = results_df[(results_df['significant']) & (results_df['regulation'] == 'Up')].index.tolist()
significant_down_genes = results_df[(results_df['significant']) & (results_df['regulation'] == 'Down')].index.tolist()

# Print significant genes for up-regulation
print("Significant genes for Up-regulation:")
print(significant_up_genes)
print(len(significant_up_genes))

# Print significant genes for down-regulation
print("\nSignificant genes for Down-regulation:")
print(significant_down_genes)
print(len(significant_down_genes))

# print("\n\nTotal Significant Genes are: ", significant_up_genes + significant_down_genes)
# print("Total Genes", len(significant_up_genes + significant_down_genes))


# Plotting
plt.figure(figsize=(12, 6))

# Scatter plot for non-significant genes
plt.scatter(results_df.loc[~results_df['significant'], 'logFC'], 
            results_df.loc[~results_df['significant'], '-log10(P.Value)'], 
            c='grey', alpha=0.5, label='Non-Significant')

# Scatter plot for significant genes - Up-regulated
plt.scatter(results_df.loc[results_df['significant'] & (results_df['regulation'] == 'Up'), 'logFC'], 
            results_df.loc[results_df['significant'] & (results_df['regulation'] == 'Up'), '-log10(P.Value)'], 
            c='#d60909', alpha=0.7, label='Significant Up-regulated')

# Scatter plot for significant genes - Down-regulated
plt.scatter(results_df.loc[results_df['significant'] & (results_df['regulation'] == 'Down'), 'logFC'], 
            results_df.loc[results_df['significant'] & (results_df['regulation'] == 'Down'), '-log10(P.Value)'], 
            c='#14abde', alpha=0.7, label='Significant Down-regulated')

# Add lines for thresholds
plt.axhline(y=-np.log10(p_value_threshold), color='blue', linestyle='--')
plt.axvline(x=logFC_threshold, color='blue', linestyle='--')
plt.axvline(x=-logFC_threshold, color='blue', linestyle='--')

# Labels and title
plt.xlabel('log2 Fold Change', fontsize=24)
plt.ylabel('-log10 P-value', fontsize=24)
plt.title('Volcano Plot')

# Add legend
plt.legend(fontsize=18)
plt.tight_layout()

# # Save the figure as a PNG file
# plt.savefig('GSE18842__Volcano_Plot_Final 02.png', dpi=500)  # Adjust dpi for higher resolution if needed

# Show the plot
plt.show()

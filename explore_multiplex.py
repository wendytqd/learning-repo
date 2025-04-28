import pandas as pd
import os
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# ---------- Load Data and Precompute Statistics ----------
# Load the data
file_path = os.path.join(os.path.dirname(__file__), 'multiplexELISA_normalized.csv')
df = pd.read_csv(file_path)

# Calculate the Global Statistics for each cytokine
cytokine_df = df.iloc[:, 5:]  # Assuming the first 5 columns are not cytokine levels

global_mean = cytokine_df.mean(skipna=True)
global_std = cytokine_df.std(skipna=True)
global_min = cytokine_df.min(skipna=True)
global_max = cytokine_df.max(skipna=True)
global_median = cytokine_df.median(skipna=True)
global_iqr = cytokine_df.quantile(0.75) - cytokine_df.quantile(0.25)
global_25 = cytokine_df.quantile(0.25)
global_75 = cytokine_df.quantile(0.75)

# Create a DataFrame to hold the global statistics
global_stats = pd.DataFrame({
    'Mean': global_mean,
    'Std': global_std,
    'Min': global_min,
    'Max': global_max,
    'Median': global_median,
    'IQR': global_iqr,
    '25th Percentile': global_25,
    '75th Percentile': global_75
})


# Calculate Subgroup-Specific Statistics on cytokine columns only
# Insert 'age_group' right after the 'age' column (as the fourth column)
age_group = pd.cut(df['age'], bins=[0, 18, 30, 40, 50, 60, 70, 80, np.inf],
                    labels=['0-18', '19-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+'],
                    right=False)
age_index = df.columns.get_loc('age') + 1
df.insert(age_index, 'age_group', age_group)
cytokine_columns = df.columns[6:]  # Assuming the first 5 columns are not cytokine levels

demographic_columns = ['gender', 'race', 'age_group']

stat_results = {}

for group_col in demographic_columns:
    stat_results[group_col] = {} # Initialize a dictionary to hold results for each demographic group
    grouped = df.groupby(group_col)

    for cytokine in cytokine_columns:
        cytokine_stats = {}
        for group_value, group_df in grouped:
            data = group_df[cytokine].dropna()
            # Calculate statistics for each group
            mean_val = data.mean()
            std_val = data.std()
            min_val = data.min()
            max_val = data.max()
            median_val = data.median()
            iqr_val = data.quantile(0.75) - data.quantile(0.25)

            cytokine_stats[group_value] = {
                'Mean': mean_val,
                'Std': std_val,
                'Min': min_val,
                'Max': max_val,
                'Median': median_val,
                'IQR': iqr_val
            }
        stat_results[group_col][cytokine] = cytokine_stats # Save the results for this cytokine

# Save everything to an Excel file
output_path = os.path.join(os.path.dirname(__file__), 'multiplexELISA_statistics.xlsx')
with pd.ExcelWriter(output_path) as writer:
    # Write global statistics to sheet 'Global'
    global_stats.to_excel(writer, sheet_name='Global')
    # Write subgroup statistics to separate sheets
    for group_col, cytokine_stats in stat_results.items():
        stats_df = pd.DataFrame(cytokine_stats) # Transpose to have cytokines as rows
        stats_df.to_excel(writer, sheet_name=group_col)

# ---------- Build Node Feature Vectors for the Directed Graph ----------

# Create a dictionary mapping cytokine names to feature vectors
node_features_dict={}

# global_stats index should be cytokine names
for cytokine in global_stats.index:
    # Extract global features as a list
    global_features = global_stats.loc[cytokine].values.tolist()

    # For each demographic group, extract the subgroup-specific features (append mean values for each group)
    demo_features = []
    for group_col in demographic_columns:
        if cytokine in stat_results[group_col]:
            group_stats = stat_results[group_col][cytokine]
            for subgroup in sorted(group_stats.keys()):
                demo_features.append(group_stats[subgroup]['Mean'])
        else:
            demo_features.append(0)

    # Concatenate global and demographic features
    feature_vector = global_features + demo_features
    node_features_dict[cytokine] = feature_vector

# ---------- Construct the Directed Graph ----------
G = nx.DiGraph()

# Add cytokine nodes with their feature vectors
for cytokine , features in node_features_dict.items():
    G.add_node(cytokine, features=features)


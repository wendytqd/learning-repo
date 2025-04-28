import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from openpyxl import load_workbook

# Load cytokine-cytokine interaction data and ELISA data
df_edges =pd.read_excel(os.path.join(os.path.dirname(__file__), 'CytokineLink_HPA_cytokine2cytokine.xlsx'))
elisa_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'multiplexELISA_normalized.csv'))
global_features_df = pd.read_excel(os.path.join(os.path.dirname(__file__), 'multiplexELISA_stat.xlsx'), sheet_name='Global', index_col=0)
gender_df = pd.read_excel(os.path.join(os.path.dirname(__file__), 'multiplexELISA_stat.xlsx'), sheet_name = 'gender')
race_df = pd.read_excel(os.path.join(os.path.dirname(__file__), 'multiplexELISA_stat.xlsx'), sheet_name = 'race')
age_group_df = pd.read_excel(os.path.join(os.path.dirname(__file__), 'multiplexELISA_stat.xlsx'), sheet_name = 'age_group')

# Filter out the cytokines that are not present in the ELISA data
elisa_cytokines = set(elisa_df.columns[5:])  # The first 5 columns are not cytokine levels

# Extract the unique cytokine names from the edges
sources = df_edges['Source_cytokine_genename'].unique()
targets = df_edges['Target_cytokine_genename'].unique()
all_cytokines_cytolink = set(sources) | set(targets)
cytokines_in_both = elisa_cytokines.intersection(all_cytokines_cytolink)

filtered_edges = df_edges[
    df_edges['Source_cytokine_genename'].isin(cytokines_in_both) &
    df_edges['Target_cytokine_genename'].isin(cytokines_in_both)
].reset_index(drop=True)

# --------Filter out the cytokines that are not present in the ELISA and CytoLink data----------
# cytokines_only_in_elisa = elisa_cytokines.difference(all_cytokines_cytolink)
# cytokines_only_in_cytolink = all_cytokines_cytolink.difference(elisa_cytokines)

# print("Cytokines in both CytoLink and ELISA:")
# for cytokine in sorted(cytokines_in_both):
#     print(cytokine)

# print('\nCytokine only in ELISA:')
# for cytokine in sorted(cytokines_only_in_elisa):
#     print(cytokine)

# print('\nCytokine only in CytokineLink:')
# for cytokine in sorted(cytokines_only_in_cytolink):
#     print(cytokine)

# Create a graph from the filtered edges
G = nx.from_pandas_edgelist(filtered_edges,
                            source = 'Source_cytokine_genename',
                            target= 'Target_cytokine_genename',
                            create_using=nx.DiGraph()
                            )


# ---------- Calculate the degree of each node ----------
degree_dict = dict(G.degree())
# Create a DataFrame to hold the degree information
degree_df = pd.DataFrame(list(degree_dict.items()), columns=['Cytokine', 'Degree'])
# Sort the DataFrame by degree
degree_df = degree_df.sort_values(by='Degree', ascending=False)
# Save the degree DataFrame to a multiplexELISA_stat file
degree_df.to_csv(os.path.join(os.path.dirname(__file__), 'node_degree.csv'), index=False)

# ---------- Build Node Feature Vectors for the Directed Graph ----------
# Create a dictionary mapping cytokine names to feature vectors
node_features_dict = {}

demographic_dataframes = {
    'gender': gender_df,
    'race': race_df,
    'age_group': age_group_df
}
demographic_columns = ['gender', 'race', 'age_group']

# global_features_df rows should be cytokine names
for cytokine in global_features_df.index:
     # Extract global features as a list
     global_features = global_features_df.loc[cytokine].values.tolist()

     # For each demographic group, extract the subgroup-specific features (append mean values for each group)
     demo_features = []
     for group_col in demographic_columns:
          df = demographic_dataframes[group_col]

          # Now filter rows where stat == 'Mean'
          df_mean = df[df['Stat'] == 'Mean']
          if cytokine in df_mean.columns:
               # Collect the mean values for each subgroup for this cytokine
               demo_features.extend(df_mean[cytokine].tolist())
          else:
               # If the cytokine is not present in this demographic group, append 0
               n_subgroups = df_mean.shape[0]
               demo_features.extend([0] * n_subgroups)             

    # Concatenate global and demographic features
     feature_vector = global_features + demo_features
     node_features_dict[cytokine] = feature_vector


# ------------ Add cytokine nodes with their feature vectors ------------
for cytokine, features in node_features_dict.items():
    if cytokine in G.nodes:
        G.nodes[cytokine]['features'] = features

# ------------ Visualise the network ------------
plt.figure(figsize=(16, 10))
pos = nx.spring_layout(G, k=7.0, iterations=200, seed=42) # positions for all nodes

nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.9)

nx.draw_networkx_edges(G, pos, arrowstyle = '->', arrowsize = 5, edge_color='gray', alpha=0.5)

nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')

plt.title('Cytokine-Cytokine Interaction Network')
plt.axis('off')
plt.show()

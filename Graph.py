import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load cytokine-cytokine interaction data and ELISA data
df_edges =pd.read_excel(os.path.join(os.path.dirname(__file__), 'CytokineLink_HPA_cytokine2cytokine.xlsx'))
elisa_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'multiplexELISA_normalized.csv'))

# Filter out the cytokines that are not present in the ELISA data
elisa_cytokines = set(elisa_df.columns[5:])  # The first 5 columns are not cytokine levels
filtered_edges = df_edges[
    df_edges['Source_cytokine_genename'].isin(elisa_cytokines) &
    df_edges['Target_cytokine_genename'].isin(elisa_cytokines)
].reset_index(drop=True)

# Create a graph from the filtered edges
G = nx.from_pandas_edgelist(filtered_edges,
                            source = 'Source_cytokine_genename',
                            target= 'Target_cytokine_genename',
                            create_using=nx.DiGraph()
                            )

# Visualise the network
plt.figure(figsize=(16, 10))
pos = nx.spring_layout(G, k=7.0, iterations=200, seed=42) # positions for all nodes

nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.9)

nx.draw_networkx_edges(G, pos, arrowstyle = '->', arrowsize = 5, edge_color='gray', alpha=0.5)

nx.draw_networkx_labels(G, pos, font_size=5, font_family='sans-serif')

plt.title('Cytokine-Cytokine Interaction Network')
plt.axis('off')
plt.show()

# Calculate the degree of each node
degree_dict = dict(G.degree())
# Create a DataFrame to hold the degree information
degree_df = pd.DataFrame(list(degree_dict.items()), columns=['Cytokine', 'Degree'])
# Sort the DataFrame by degree
degree_df = degree_df.sort_values(by='Degree', ascending=False)
# Save the degree DataFrame to a CSV file
degree_df.to_csv(os.path.join(os.path.dirname(__file__), 'cytokine_degrees.csv'), index=False)
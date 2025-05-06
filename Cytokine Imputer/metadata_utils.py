import os
import pandas as pd
import numpy as np
from graph_utils import data

# Load the datasets
elisa_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "multiplexELISA_normalized.csv"))

race_to_idx = {r: i for i, r in enumerate(data.race.unique())}
gender_to_idx = {g: i for i, g in enumerate(data.gender.unique())}

n_race = len(race_to_idx)
n_gender = len(gender_to_idx)


def encode_metadata(row):
    """Encode the metadata for a single row."""
    race_vec = np.eye(n_race)[race_to_idx[row.race]]
    gender_vec = np.eye(n_gender)[gender_to_idx[row.gender]]
    age = int(row.age)
    return np.concatenate([race_vec, gender_vec, [age]])  # Concatenate the features

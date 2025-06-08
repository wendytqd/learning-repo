import os
import pandas as pd
import numpy as np
from graph_utils import data

# Load the datasets
elisa_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data/multiplexELISA_normalized.csv"))

race_to_idx = {r: i for i, r in enumerate(data.race.unique())}
gender_to_idx = {g: i for i, g in enumerate(data.gender.unique())}


n_race = len(race_to_idx)
n_gender = len(gender_to_idx)

mean_age = data.age.mean()
std_age = data.age.std() if data.age.std() > 0 else 1.0


def encode_metadata(row):
    """Encode the metadata for a single row."""
    race_vec = np.eye(n_race)[race_to_idx[row.race]]
    gender_vec = np.eye(n_gender)[gender_to_idx[row.gender]]
    age = float(row.age)
    age_norm = (age - mean_age) / std_age
    return np.concatenate([race_vec, gender_vec, [age_norm]])  # Concatenate the features

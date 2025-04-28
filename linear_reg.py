import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load the data
file_path = os.path.join(os.path.dirname(__file__), 'multiplexELISA_normalized.csv')
df = pd.read_csv(file_path)

# 2. Choose the target cytokine
cytokine = df.columns[5:]  # Assuming the first 5 columns are not cytokine levels
target = 'IL6'
features = [c for c in cytokine if c!= target] # Exclude the target cytokine

# 3. Simulate masking for evaluation
# randomly mask 20% of the target cytokine values
mask_idx = df[df[target].notnull()].sample(frac=0.2, random_state=42).index
df_masked = df.copy() 
df_masked.loc[mask_idx, target] = np.nan # Simulate missing values


# 4. Split the data into training and test sets
train_df = df_masked[df_masked[target].notnull()]

# Only rows we explicitly masked will be used for testing
pred_mask = df_masked[target].isna() & df[target].notna()
test_df = df_masked[pred_mask]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]


# 5. Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# 6. Predict the missing values
Y_test = rf.predict(X_test)

# 7. Put predictions back in the original DataFrame
df_masked.loc[test_df.index, target] = Y_test

# 8. Evaluate the model
y_true = df.loc[mask_idx, target]
mse = mean_squared_error(y_true, Y_test)
rmse = np.sqrt(mse)
mean_val = df[target].mean()
relative_error = rmse / mean_val
print(f'Root Mean Squared Error: {rmse:.3f}')
print('Mean concentration:', mean_val)
print("relative error:", relative_error)

# Histogram of target cytokine (IL-6)
plt.hist(df[target].dropna(), bins=30, edgecolor='black')
plt.title(f'{target} Distribution')
plt.show()

# Prediction vs True values
plt.scatter(y_true, Y_test, alpha=0.5)
plt.plot([0, max(y_true)], [0, max(y_true)], 'r--')  # Diagonal line
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'Predicted vs True Values for {target}')
plt.show()

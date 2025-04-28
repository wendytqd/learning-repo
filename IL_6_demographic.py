import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'multiplexELISA_normalized.csv')
df = pd.read_csv(data_path)

# 2. Select the target and demographic features
target = 'IL6'
demo_features = ['age', 'gender', 'race']

# Drop row where IL6 is missin
df_dem = df[df[target].notnull()][demo_features + [target]].dropna()

X = df_dem[demo_features]
Y = df_dem[target]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. Build a preprocessing _ model pipeline
# Scale age
# One-hot encode gender and race
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age']),
    ('cat', OneHotEncoder(drop='first'), ['gender', 'race'])
])

model = Pipeline([
    ('prep', preprocessor),
    ('reg', Ridge(alpha=1.0))
])

# 5. Cross-validate on the training set
cv_scores = cross_val_score(model, X_train, y_train,
                            scoring='neg_root_mean_squared_error',
                            cv=5)
print(f'Cross-validated RMSE: {-np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')

# 6. Fit the model on the training set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Evaluate the model on the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'Test RMSE: {rmse:.2f}')
print(f'Test R²: {r2:.2f}')


# 8. Predicted vs. actual Il6 by gender
results = X_test.copy().reset_index(drop=True)
results['actual'] = y_test.reset_index(drop=True)
results['predicted'] = y_pred
results['residual'] = results['predicted'] - results['actual']

plt.figure(figsize=(8, 8))
ax = sns.scatterplot(data=results, x='actual', y='predicted', hue='gender', edgecolor='w')
sns.regplot(data=results, x='actual', y='predicted', scatter=False, ax=ax, line_kws={'color':'black', 'lw':'1'})
plt.xlabel('Actual IL6 (pg/mL)')
plt.ylabel('Predicted IL6 (pg/mL)')
plt.title('Predicted vs. Actual IL-6 by Gender')
plt.legend(title= 'Gender')
plt.show()

# 9. Predicted vs. Actual IL-6 by race
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=results, x='actual', y='predicted', hue = 'race', edgecolor='w')
sns.regplot(data=results, x='actual', y='predicted', scatter=False, ax=ax, line_kws={'color':'black', 'lw':'1'})
plt.xlabel('Actual IL6 (pg/mL)')
plt.ylabel('Predicted IL6 (pg/mL)')
plt.title('Predicted vs. Actual IL-6 by Race')
plt.legend(title = 'Race')
plt.show()

# 10. Residuals vs age by gender
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=results, x='age', y='residual', hue = 'gender', edgecolor='w')
sns.regplot(data=results, x='age', y='residual', scatter=False, ax=ax, line_kws={'color':'black', 'lw':'1'})
plt.axhline(0, color='k', linestyle='--', lw=1)
plt.xlabel('Age')
plt.ylabel('Residual (Predicted - Actual)')
plt.title('Residuals of IL-6 Predictions vs. Age by Gender')
plt.legend(title = 'Gender')
plt.show()

# 11. Residuals by Race (Boxplot)
plt.figure(figsize=(8, 4))
sns.boxplot(data=results, x='race', y='residual')
plt.axhline(0, color='k', linestyle='--', lw=1)
plt.title('Residuals of IL-6 Preidction by Race')
plt.xlabel('Race')
plt.ylabel('Residual (Predicted - Actual)')
plt.xticks(rotation=45)
plt.show()
# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Read the data
data = pd.read_csv('data/cardio_train.csv')

# Select subset of predictors
cols_to_use = ['age', 'gender', 'weight', 'height', 'alco', 'smoke', 'active']
X = data[cols_to_use]

# Select target
y = data.cardio

# %%

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

# %%

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

# %%
print("Average MAE score (across experiments):")
print(scores.mean())
# %%

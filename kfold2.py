# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble, svm

# %%
train_data = pd.read_csv('data/cardio_train.csv')

# Remove rows with missing target values
train_data.dropna(axis=0, subset=['cardio'], inplace=True)
y = train_data.cardio # Target variable             
train_data.drop(['cardio'], axis=1, inplace=True) # Removing target variable from training data

train_data.drop(['ap_lo', 'ap_hi', 'gluc', 'cholesterol', 'id'], axis=1, inplace=True) # Remove columns with null values

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()

print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))

X.head() # Show first 5 training examples

# %% 
clf = linear_model.LogisticRegression(random_state=0).fit(X, y)
scores = cross_val_score(clf, X, y, cv=5)
scores

# %% 
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# %%
# Lets split the data into 5 folds.  
# We will use this 'kf'(KFold splitting stratergy) object as input to cross_val_score() method
kf =KFold(n_splits=5, shuffle=True, random_state=42)

cnt = 1
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(X, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1
# %%
def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')
# %% using LR
score = cross_val_score(linear_model.LinearRegression(), X, y, cv= kf, scoring="neg_mean_squared_error")
print(f'Scores for each fold: {score}')
rmse(score.mean())


# %%
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))
# %% using DT 
score = cross_val_score(tree.DecisionTreeRegressor(random_state= 42), X, y, cv=kf, scoring="neg_mean_squared_error")
print(f'Scores for each fold: {score}')
rmse(score.mean())

# %%
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))
# %% using RF
score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")
print(f'Scores for each fold are: {score}')
rmse(score.mean())

# %% 
print("%0.2f accuracy with a standard deviation of %0.2f" % (score.mean(), score.std()))
# %%
max_depth = [1,2,3,4,5,6,7,8,9,10]

for val in max_depth:
    score = cross_val_score(tree.DecisionTreeRegressor(max_depth= val, random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")
    print(f'For max depth: {val}')
    rmse(score.mean())

# %% 
estimators = [50, 100, 150, 200, 250, 300, 350]

for count in estimators:
    score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= count, random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")
    print(f'For estimators: {count}')
    rmse(score.mean())

# %%
max_depth = [1,2,3,4,5,6]

for val in max_depth:
    score = cross_val_score(tree.DecisionTreeRegressor(max_depth= val, random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")
    print(f'For max depth: {val}')
    rmse(score.mean())
# %%
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
scores

# %%
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
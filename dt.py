# %%
import pandas as pd
import numpy as np
from sklearn import linear_model, tree, ensemble, svm
from sklearn.tree import DecisionTreeClassifier 
import graphviz
from sklearn import tree
from interpret.glassbox import ClassificationTree
from dtreeviz.trees import dtreeviz

# %%
train_data = pd.read_csv('data/cardio_train.csv')

print(train_data.keys())
train_data.all

# %% 
X = pd.DataFrame(train_data.data)
X.columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']
Y= pd.DataFrame(train_data.target)
Y.columns = ['cardio']

# %% Remove rows with missing target values
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
clf = DecisionTreeClassifier(random_state=1234)
clf = pd.DataFrame()

# %% 
# tree = ClassificationTree()
# tree.fit(X, y)
# print("Training finished.")

# %%
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=train_data.feature_names,  
                                class_names=train_data.target_names,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph
# %%

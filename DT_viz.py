# %%
from re import X
from turtle import color
import pip
from sklearn.datasets import make_blobs
from utils import DataLoader
import pandas  
from interpret.glassbox import (LogisticRegression,
                                ClassificationTree, 
                                ExplainableBoostingClassifier)
from interpret import show
from sklearn.metrics import f1_score, accuracy_score
#import sklearn.external.joblib as extjoblib
import joblib
from matplotlib import pyplot
from numpy import where 
from numpy import meshgrid
from numpy import arange
from numpy import hstack
from interpret import perf
from sklearn.datasets import *
from sklearn import tree
from dtreeviz.trees import *
import numpy as np
# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
# X_train, X_test, y_train, y_test = data_loader.get_data_split()
# print(X_train.shape)
# print(X_test.shape)
# Oversample the train data
# X_train, y_train = data_loader.oversample(X_train, y_train)
# print("After oversampling:", X_train.shape)
# %% Fit decision tree model
# tree = ClassificationTree()
# tree.fit(X_train, y_train)
# print("Training finished.")
# y_pred = tree.predict(X_test)
# print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
# print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain local prediction
# tree_local = tree.explain_local(X_test[:100], y_test[:100], name='Tree')
# show(tree_local)

# %% Explain global tree prediction 
# tree_global = tree.explain_global(name='Tree')
# show(tree_global)

# %% documentation https://github.com/parrt/dtreeviz 

regr = tree.DecisionTreeRegressor(max_depth=2)
cvd = pandas.read_csv('data/cardio_train.csv')()
regr.fit(cvd.data, cvd.target)

viz = dtreeviz(regr,
               cvd.data,
               cvd.target,
               target_name='cardio',
               feature_names=cvd.feature_names)
              
viz.view()

#%%

classifier = tree.DecisionTreeClassifier(max_depth=2)  # limit depth of tree
iris = data_loader()
classifier.fit(iris.data, iris.target)

viz = dtreeviz(classifier, 
               iris.data, 
               iris.target,
               target_name='cardio',
               feature_names=iris.feature_names, 
               class_names=["1", "0"]  # need class_names for classifier
              )  
              
viz.view()

# %%

regr = tree.DecisionTreeRegressor(max_depth=2)  # limit depth of tree
diabetes = data_loader()
regr.fit(diabetes.data, diabetes.target)
X = diabetes.data[np.random.randint(0, len(diabetes.data)),:]  # random sample from training

viz = dtreeviz(regr,
               diabetes.data, 
               diabetes.target, 
               target_name='value', 
               orientation ='LR',  # left-right orientation
               feature_names=diabetes.feature_names,
               X=X)  # need to give single observation for prediction
              
viz.view() 
# %%

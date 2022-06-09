# %% Imports
from re import X
import numpy
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from interpret.blackbox import LimeTabular
from interpret import show
import joblib
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import where 
from numpy import meshgrid
from numpy import arange
from numpy import hstack
import seaborn as sns
import time
from random import randint
from algorithms.sort import quick_sort


startTime = time.time()

# Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

# Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

lime = LimeTabular(predict_fn=rf.predict_proba, 
                   data=X_train, 
                   random_state=1)
# Get local explanations
# lime_local = lime.explain_local(X_test[-50:], 
#                                 y_test[-50:], 
#                                 name='LIME')

lime_local = lime.explain_local(X_test[-1:], 
                                y_test[-1:], 
                                name='LIME')

show(lime_local)

# executionTime = (time.time() - startTime)
# print('Execution time in seconds: ' + str(executionTime))

# %%
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import shap
import joblib
import time
from sklearn.model_selection import cross_val_score

startTime = time.time()

# Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
# print(f"Accuracy {accuracy_score(y_test, y_pred)}")

lr = LogisticRegression(random_state=2021, feature_names=X_train.columns, penalty='l1', solver='liblinear')
lr.fit(X_train, y_train)
print("Training finished.")

# %% Evaluate logistic regression model
y_pred = lr.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

explainer = shap.TreeExplainer(rf)
# Calculate shapley values for test data
start_index = 1
end_index = 2
shap_values = explainer.shap_values(X_test[start_index:end_index])
X_test[start_index:end_index]

print(shap_values[0].shape)
shap_values

shap.initjs()
# Force plot
prediction = rf.predict(X_test[start_index:end_index])[0]
print(f"The RF predicted: {prediction}")
shap.force_plot(explainer.expected_value[1],
                shap_values[1],
                X_test[start_index:end_index])

# executionTime = (time.time() - startTime)
# print('Execution time in seconds: ' + str(executionTime))

               
# %%


# %%

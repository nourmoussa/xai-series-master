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
from numpy import where 
from numpy import meshgrid
from numpy import arange
from numpy import hstack
import seaborn as sns

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print(X_train.shape)
print(X_test.shape)

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")


# %% 
# X_train = numpy.arange(min(X_train), max(X_train), 0.01)
# X_train = X_train.reshape((len(X_train), 1))
# pyplot.scatter(X_train, y_train, color='r')
# pyplot.plot(X_train, rf.predict(X_train), color='blue')
# pyplot.show()
# data = sns.load_dataset()
# sns.regplot(X_train, y_train, lowess=True)

# %% Apply lime
# Initilize Lime for Tabular data
lime = LimeTabular(predict_fn=rf.predict_proba, 
                   data=X_train, 
                   random_state=1)
# Get local explanations
lime_local = lime.explain_local(X_test[-50:], 
                                y_test[-50:], 
                                name='LIME')

show(lime_local)

# %%
# j = 0
# categorical_features = numpy.argwhere(numpy.array(len(set(X_train.values[:,X_train]))))
# for x in range((X_train.values.shape[1]) <= 10).flatten():
#     # LIME has one explainer for all models
#     explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
#         feature_names=X_train.columns.values.tolist(),
#         class_names=['price'],
#         categorical_features=categorical_features,
#         verbose=True, mode='regression')
# %%        
exp = lime.explain_instance(X_test.values[j], rf.predict, num_features=5)
exp.show_in_notebook(show_table=True)

# %% Get global explanations 

lime_global = lime.explain_global()

show(lime_global)

# %% job lib save  - LIME
joblib.dump(lime, 'lime.pkl')

joblib_model= joblib.load('lime.pkl')
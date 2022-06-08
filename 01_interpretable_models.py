
# %% Imports
from re import X
from turtle import color
import pip
from sklearn.datasets import make_blobs
from utils import DataLoader
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
from dtreeviz.trees import dtreeviz 
# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
print(X_train.shape)
print(X_test.shape)
# Oversample the train data
# X_train, y_train = data_loader.oversample(X_train, y_train)
# print("After oversampling:", X_train.shape)

# %% Fit logistic regression model
lr = LogisticRegression(random_state=2021, feature_names=X_train.columns, penalty='l1', solver='liblinear')
lr.fit(X_train, y_train)
print("Training finished.")

# %% Evaluate logistic regression model
y_pred = lr.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain local prediction
lr_local = lr.explain_local(X_test[:100], y_test[:100], name='Logistic Regression')
show(lr_local)

# %% lr local 
joblib.dump(lr, 'lr_local.pkl')

joblib_model= joblib.load('lr_local.pkl')

# %% Explain global logistic regression model
lr_global = lr.explain_global(name='Logistic Regression')
show(lr_global)

# %% job lib save  - Regression model 
joblib.dump(lr, 'reg_1.pkl')

joblib_model= joblib.load('reg_1.pkl')

# %% Fit decision tree model
tree = ClassificationTree()
tree.fit(X_train, y_train)
print("Training finished.")
y_pred = tree.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain local prediction
tree_local = tree.explain_local(X_test[:100], y_test[:100], name='Tree')
show(tree_local)

# %% Explain global tree prediction 
tree_global = tree.explain_global(name='Tree')
show(tree_global)

# %% job lib save  - Regression model 
joblib.dump(tree, 'tree.joblib')

joblib_model= joblib.load('tree.joblib')

# %% Fit Explainable Boosting Machine
ebm = ExplainableBoostingClassifier(random_state=2021)
ebm.fit(X_train, y_train) 
print("Training finished.")
y_pred = ebm.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Explain locally
ebm_local = ebm.explain_local(X_test[:100], y_test[:100], name='EBM')
show(ebm_local)

# %% Explain globally
ebm_global = ebm.explain_global(name='EBM')
show(ebm_global)

# %% save the file as joblib
import joblib

from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors = 3)

knn.fit(X_train, y_train)

print("Training finished.")

y_pred = knn.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# %% Save the model as a pickle in a file
joblib.dump(knn, 'knn.joblib')
 
# Load the model from the file
knn_from_joblib = joblib.load('knn.joblib')
 
# Use the loaded model to make predictions
knn_from_joblib.predict(X_test)

# %% 
from sklearn import svm

from sklearn.model_selection import GridSearchCV

#Create a svm Classifier and hyper parameter tuning 
ml = svm.SVC() 
  
# defining parameter range
param_grid = {'C': [ 1, 10, 100, 1000,10000], 
              'gamma': [1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(ml, param_grid, refit = True, verbose = 1,cv=5)
  
# fitting the model for grid search
grid_search=grid.fit(X_train, y_train)

accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )


# %% classifir plot 
#X_train, y_train = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
for class_value in range(2): 
    row_ix = where(y_train == class_value)
    pyplot.scatter(X_train[row_ix, 0], X_train[row_ix,1])
pyplot.show()



# %%
# define bounds of the domain
min1, max1 = X_train[:, 0].min()-1, X_train[:, 0].max()+1
min2, max2 = X_train[:, 1].min()-1, X_train[:, 1].max()+1
# define the x and y scale
x1grid = arange(min1, max1, 0.1)
x2grid = arange(min2, max2, 0.1)
# create all of the lines and rows of the grid
xx, yy = meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = hstack((r1,r2))
# define the model
model = LogisticRegression()
# fit the model
model.fit(X_train, y_train)
# make predictions for the grid
yhat = model.predict(grid)
# reshape the predictions back into a grid
zz = yhat.reshape(xx.shape)
# plot the grid of x, y and z values as a surface
pyplot.contourf(xx, yy, zz, cmap='Paired')

print('zz')

# create scatter plot for samples from each class
# highlight = X_test[418:422]
for class_value in range(2):
	# get row indexes for samples with this class
	row_ix = where(y_train == class_value)
	# create scatter of these samples
	pyplot.scatter(X_train[row_ix, 0], X_train[row_ix, 1], cmap='Paired')
pyplot.plot(-5,0, marker='o',color='r') 

pyplot.show()

#%%
pyplot.savefig("decision_surface.png")

# %% 

show([lr_global, tree_global, ebm_global])

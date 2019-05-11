import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')

# load built in iris dataset
iris = sns.load_dataset('iris')
iris.head()

# create pairplot of iris dataset labeled by species
sns.pairplot(iris,hue = 'species')

# seperate data by species
versicolor = iris[iris['species']=='versicolor']
virginica = iris[iris['species']=='virginica']
setosa = iris[iris['species']=='setosa']

# plot kde plot of sepal length vs sepal width of setosa species
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap = 'plasma', shade = True, shade_lowest = False)

# seperate data into training set and test set
from sklearn.model_selection import train_test_split
X = iris.drop('species', axis = 1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create and fit SVC model
from sklearn.svm import SVC
model = SVC(gamma = 'scale')
model.fit(X_train, y_train)

# evaluate SVC model
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# use grid search to find best parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 5)
grid.fit(X_train, y_train)
grid.best_params_

# evaluate grid and compare with previous SVC model
grid_pred = grid.predict(X_test)
print(confusion_matrix(y_test, grid_pred))
print(classification_report(y_test,grid_pred))

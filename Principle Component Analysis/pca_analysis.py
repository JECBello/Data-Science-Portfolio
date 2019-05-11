import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')

# load breast cancer data set, view features and target
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
df.head()
cancer['target']

# Scale data prior to doing analysis
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_df = scaler.transform(df)

# import PCA and transform data 
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca.fit(scaled_df)
x_pca = pca.transform(scaled_df)

# plot scatterplot of first two principle components, colored by the third principle component
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1], c=x_pca[:,2],cmap='magma')
plt.xlabel('First Principle Component')
plt.ylabel('Second Principle Component')

# create heatmap to visualize which features are most responsible for principle components
df_comp = pd.DataFrame(pca.components_,columns = cancer['feature_names'])
plt.figure(figsize=(12,6))
sns.heatmap(df_comp, cmap = 'magma')

# Use machine learning algorithms on data
# seperate data into training set and testing set
from sklearn.model_selection import train_test_split
X = x_pca
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)

# create and fit Logistic Regression model, then use model to predict test set results 
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
logreg_pred = log_model.predict(X_test)

# evaluate model 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,logreg_pred))
print('\n')
print(classification_report(y_test, logreg_pred))

# create and fit SVC. Then use model to predict test set results
from sklearn.svm import SVC
svc_model = SVC(gamma = 'scale')
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)

# evaluate model
print(confusion_matrix(y_test,svc_pred))
print('\n')
print(classification_report(y_test, svc_pred))


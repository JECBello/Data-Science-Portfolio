import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')

# Read CSV file and check contents
df = pd.read_csv('KNN_Project_Data')
df.head()

# Create pairplot using hue = TARGET CLASS
sns.pairplot(df, hue = 'TARGET CLASS', palette = 'RdBu_r')

# Standardize variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis = 1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))
df_scaled = pd.DataFrame(scaled_features, columns = df.columns[:-1])
df_scaled.head()

# split data into training set and testing set
from sklearn.model_selection import train_test_split
X = df_scaled
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

# Using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(1)
knn.fit(X_train, y_train)

# Evaluate model 
from sklearn.metrics import classification_report, confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Choose a better K value using elbow method
error_list = []

for i in range(1,40):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_list.append(np.mean(pred_i != y_test))

# Plot graph of error and determine which K value resulted in least error rate
plt.figure(figsize = (10,6))
plt.plot(range(1,40), error_list, color='blue', ls = '--',marker = 'o',
        markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate') 

# Retrain using new K value
knn = KNeighborsClassifier(29)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
print('WITH K=29')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

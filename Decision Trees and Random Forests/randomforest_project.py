import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')

# read loan data
loans = pd.read_csv('loan_data.csv')
loans.info()
loans.describe()
loans.head()

# plot histogram of FICO distribution based on credit policy
hist1 = loans['fico'][loans['credit.policy'] == 1]
hist2 = loans['fico'][loans['credit.policy'] == 0]

plt.figure(figsize=(10,6))
plt.hist(hist1, bins = 30, alpha = 0.5, label = 'Credit Policy = 1')
plt.hist(hist2, bins = 30, alpha = 0.5, label = 'Credit Policy = 0')
plt.xlabel('FICO')
plt.legend()

# plot histogram of FICO distribution based on whether borrower has paid or not
paid1 = loans['fico'][loans['not.fully.paid'] == 1]
paid2 = loans['fico'][loans['not.fully.paid'] == 0]

plt.figure(figsize=(10,6))
plt.hist(paid1, bins = 30, alpha = 0.5, label = 'not.fully.paid = 1')
plt.hist(paid2, bins = 30, alpha = 0.5, label = 'not.fully.paid = 0')
plt.xlabel('FICO')
plt.legend()

# plot countplot of purpose of loans seperated by whether the loan was fully paid back or not
plt.figure(figsize=(10,6))
sns.countplot('purpose', hue = 'not.fully.paid', data = loans)

# create jointplot of interest rate vs FICO score
plt.figure(figsize = (10,6))
sns.jointplot('fico', 'int.rate', loans, xlim = (600,850), ylim = (0,0.25), 
             joint_kws = {'s':15})

# Plot linear model to visualize trend difference between not.fully.paid and credit policy columns
lm = sns.lmplot('fico','int.rate', loans, hue = 'credit.policy', col = 'not.fully.paid')
axes = lm.axes
axes[0,0].set_xlim([550,850])
axes[0,0].set_ylim([0,0.25])

# create dummy variables for categorical data
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)

# seperate data into training set and testing set
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

# Instantiate decision tree classifier, train, and predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

# view metrics
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# Instantiate random forest classifier, train, and predict
from sklearn.ensemble import RandomForestClassifier
rand_tree = RandomForestClassifier(n_estimators = 200)
rand_tree.fit(X_train, y_train)
rand_preds = rand_tree.predict(X_test) 

# view metrics and compare with decision tree metrics 
print(classification_report(y_test, rand_preds))
print(confusion_matrix(y_test,rand_preds))

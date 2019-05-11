import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')

# Read advertising data
ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()

# Plot histogram of Age column
sns.distplot(ad_data['Age'], bins = 30, kde = False, color = 'red')

# Plot jointplot of Area Income vs. Age
sns.jointplot('Age', 'Area Income', ad_data,
              marginal_kws = dict(bins = 40),
              joint_kws = dict(s = 15))
              
# Create jointplot showing kde distributions of Daily Time spent on site vs. Age
sns.jointplot('Age', 'Daily Time Spent on Site', ad_data, kind = 'kde', color = 'red')

# Create jointplot of Daily Time Spent on Site vs. Daily Internet Usage
sns.jointplot('Daily Time Spent on Site', 'Daily Internet Usage', ad_data, color = 'green')

# Create pairplot with hue defined by Clicked on Ad column
sns.pairplot(ad_data, hue = 'Clicked on Ad', palette = 'RdBu_r')

# Prepare Data 
city = pd.get_dummies(ad_data['City'], drop_first = True)
country = pd.get_dummies(ad_data['Country'], drop_first = True)
ad_data.drop(['Ad Topic Line', 'Timestamp'], axis = 1, inplace = True)
X = pd.concat([ad_data, city, country], axis = 1)
y = ad_data['Clicked on Ad']

# import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

# import logistic regression model, fit, and predict results
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

# Evaluate results
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

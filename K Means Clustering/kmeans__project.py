import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')

# read csv file containing data
college = pd.read_csv('College_Data',index_col = 0)
college.head()
college.info()
college.describe()

# create scatterplot comparing graduation rate vs room board, seperated by whether school was private
plt.figure(figsize=(7,7))
sns.scatterplot('Room.Board', 'Grad.Rate','Private',data = college, palette = 'RdBu_r')

# create similar graph but instead comparing fulltime undergraduates to out-of-state tuition
plt.figure(figsize=(7,7))
sns.scatterplot('Outstate','F.Undergrad','Private',data = college, palette = 'RdBu_r')

# plot histogram of out-of-state tuition, seperated by whether the univeristy was private or not
sns.set_style('darkgrid')
plt.figure(figsize=(10,6))
g = sns.FacetGrid(college, hue = 'Private', legend_out = True, height = 6, aspect = 1.75)
g.map(plt.hist, 'Outstate', bins = 20, alpha = 0.5)

# plot histogram of graduation rate, separated by whether university was private or not
plt.figure(figsize=(10,6))
g = sns.FacetGrid(college, hue = 'Private', legend_out = True, height = 6, aspect = 1.75)
g.map(plt.hist, 'Grad.Rate', bins = 20, alpha = 0.5)

# the graduation rate of this college was determined to be greater than 100 from histograms, need to be adjusted
college.loc['Cazenovia College', 'Grad.Rate'] = 100

# create K means clustering model and fit to college data
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(college.drop('Private', axis = 1))
kmeans.cluster_centers_
kmeans.labels_

# Realistically k means clustering is unsupervised so it cannot be evaluated, however for learning purposes 
# the results will be evaluated using the 'Private' column of the dataset

# Create new column for college converting 'Private' column to boolean value for evaluation
college['cluster'] = pd.Series(np.where(college['Private'] == 'Yes',1,0),
                               college.index) 

# View classification report and confustion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(college['cluster'], kmeans.labels_))
print(classification_report(college['cluster'], kmeans.labels_))

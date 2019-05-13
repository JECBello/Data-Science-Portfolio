import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# read yelp review data
yelp = pd.read_csv('yelp.csv')
#yelp.head()
#yelp.info()
#yelp.describe()

import string
#create function to count number of words in review and apply to text column. Set output to new col 'text_length'
def count_words(review):
    '''
    1.Remove all punctuation from reviews
    2.use .split() to seperate reviews into a list
    3.use len function to count number of words in each list
    '''
    
    no_punc = [char for char in review if char not in string.punctuation]
    txt_review = ''.join(no_punc)
    words_list = txt_review.split()
    return len(words_list)
    
yelp['text_length'] = yelp['text'].apply(count_words)

# data visualization
fg = sns.FacetGrid(yelp,col='stars')
fg.map(plt.hist, 'text_length')
sns.boxplot(x ='stars', y='text_length', data = yelp)
sns.countplot('stars', data = yelp)

df_mean = yelp.groupby(yelp['stars']).mean()
df_corr = df_mean.corr()
sns.heatmap(df_mean.corr(), annot = True, cmap = 'magma')

# for classification task, only reviews that were either a 1 star or 5 star were considered
# classification
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create pipieline process, scoores are displayed in comments next to classifier used
process = Pipeline([
    ('Vectorize', CountVectorizer()),
    #('TfId', TfidfTransformer()),
    ('NB',MultinomialNB()) # 0.93/0.93/0.92
    #('SVC', SVC()) #Bad
    #('DTree', DecisionTreeClassifier(n_estimators=100)) # 0.93/0.93/0.92
    #('Random_Forest', RandomForestClassifier()) # 0.85 / 0.86 / 0.84
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)

# evaluate models
from sklearn.metrics import classification_report,confusion_matrix
process.fit(X_train,y_train)
pipe_predictions = process.predict(X_test)
print(confusion_matrix(y_test,pipe_predictions))
print('\n')
print(classification_report(y_test,pipe_predictions))



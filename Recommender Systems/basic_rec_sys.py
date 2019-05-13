import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('whitegrid')

# create dataframe from data
colum_names = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('u.data', sep='\t', names = colum_names)
movie_titles = pd.read_csv('Movie_Id_Titles')
df = pd.merge(df, movie_titles, on='item_id')

# display movies by rating in descending order
#df.groupby('title')['rating'].mean().sort_values(ascending = False).head()
# display movies by number of ratings in descending order
#df.groupby('title')['rating'].count().sort_values(ascending = False).head()

# create dataframe representing the ratings of each movie, along with the number of ratings the movie has
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

# visualize ratings based on number and value
ratings['num of ratings'].hist(bins = 70)
ratings['rating'].hist(bins = 70)
sns.jointplot(x='rating',y = 'num of ratings', data = ratings, alpha = 0.5)

# create pivot table displaying what each user rated each movie
moviemat = df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')

# Choose which movies to base recommender system on
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

# Create correlation table comparing average star wars/ liar liar ratings to other movies
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

# Create dataframe displaying correlation and the number of ratings  
corr_starwars = pd.DataFrame(similar_to_starwars, columns = ['Correlation'])
corr_starwars.dropna(inplace = True)
corr_starwars = corr_starwars.join(ratings['num of ratings'])

# Only recommend movies with high correlations and a greater number of reviews than 100
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',
                                                              ascending = False).head()
                                                              

# repeat above process with liar liar movie
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns = ['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',
                                                              ascending = False).head()


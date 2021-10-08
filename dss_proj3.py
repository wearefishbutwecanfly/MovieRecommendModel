# -*- coding: utf-8 -*-
"""DSS_Proj3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L9uQ3XMMxNukQJlLh5BCtTucsJ15M-an
"""

# Commented out IPython magic to ensure Python compatibility.
# Installation
# pip install numpy
# pip install surprise
# pip install -U scikit-learn
# pip install pandas
# print("-------------INSTALL SURPRISE COMPLETE-------------------")
# %matplotlib inline
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
print("-------------IMPORT COMPLETE-------------")

smd = pd.read_csv('/content/drive/MyDrive/input/smd.csv')
smd.head(3)

# Xử lí data
smd['genres'] = smd['genres'].apply(literal_eval)

"""# Recommend based on Weighted Rating"""

vote_counts = smd[smd['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = smd[smd['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C

m = vote_counts.quantile(0.95)
m

"""# Recommend not based on Genres"""

qualified = smd[(smd['vote_count'] >= m) & (smd['vote_count'].notnull()) & (smd['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified['wr'] = qualified.apply(weighted_rating, axis=1)

qualified = qualified.sort_values('wr', ascending=False).head(250)
qualified.head(15)

"""# Recommend based on Genres"""

s = smd.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = smd.drop('genres', axis=1).join(s)

s.head()
# gen_md.head(10)

def build_chart(genre, percentile=0.95):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified

build_chart('Romance').head(15)
# df1 = build_chart('Romance').head(15)
# df2 = build_chart('Romance', percentile = 0.85).head(15)
# df3 = pd.concat([df1, df2], axis=1)
# df3



"""# Content Based Recommender"""

count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

get_recommendations('The Dark Knight').head(10)



"""# Hybrid Recommender"""

reader = Reader()
ratings = pd.read_csv('/content/drive/MyDrive/archive/ratings_small.csv')
# ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv = 5, verbose=0)
trainset = data.build_full_trainset()
svd.fit(trainset)
svd.predict(1, 302,3)

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

id_map = pd.read_csv('/content/drive/MyDrive/archive/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')

def hybrid(userId, title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)

hybrid(1, 'Avatar')



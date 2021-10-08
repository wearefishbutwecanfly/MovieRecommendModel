from CSV_load import *

print("START TRAINING")
#Cosine sim setting
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
# Hybrid Recommender
print('25%')
indices = pd.Series(smd.index, index=smd['title'])
reader = Reader()
# ratings.head()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
print('50%')
# cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=0)
trainset = data.build_full_trainset()
print('75%')
svd.fit(trainset)
print('100%')
# svd.predict(1, 302, 3)
print("FITTING COMPLETE")

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

# print(hybrid(1, 'Avatar'))

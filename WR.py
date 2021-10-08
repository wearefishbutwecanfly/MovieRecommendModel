from CSV_load import *

"""# Recommend based on Weighted Rating"""

#Calculate Vote Average
vote_counts = smd[smd['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = smd[smd['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

#Calculate Vote Quantile
m = vote_counts.quantile(0.95)

#Weighted Rating Function
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

"""# Recommend not based on Genres"""

qualified = smd[(smd['vote_count'] >= m) & (smd['vote_count'].notnull()) & (smd['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
# qualified.shape

qualified['wr'] = qualified.apply(weighted_rating, axis=1)

qualified = qualified.sort_values('wr', ascending=False).head(250)
# print(qualified.head(15))
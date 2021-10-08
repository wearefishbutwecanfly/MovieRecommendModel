# Weight Rating Recommendation
from WR import *
print(qualified.head(30))

# Genres Recommendation
from Genres import *
print(build_chart('Family').head(15))

# Content Recommendation

from Content import *
print(get_recommendations('Assassins').head(10))

#Hybrid Recommendation
from Hybrid import hybrid
print(hybrid(1, 'Toy Story'))
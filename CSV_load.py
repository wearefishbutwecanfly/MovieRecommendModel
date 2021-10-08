# Installation
# pip install numpy
# pip install surprise
# pip install -U scikit-learn
# pip install pandas
# print("-------------INSTALL SURPRISE COMPLETE-------------------")

import pickle
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
print("-------------IMPORT COMPLETE-------------")

smd = pickle.load(open('input/smd.pkl', 'rb'))
ratings = pickle.load(open('input/ratings.pkl', 'rb'))
indices_map = pickle.load(open('input/indices_map.pkl', 'rb'))

print("-------------IMPORT CSV COMPLETE-------------")
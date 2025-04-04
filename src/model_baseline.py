import pandas as pd
from collections import Counter

class PopularityRecommender:
    def __init__(self):
        self.popular_movies = []

    def fit(self, df: pd.DataFrame):
        self.popular_movies = df['movieId'].value_counts().index.tolist()

    def recommend(self, user_id: int, k=20):
        return self.popular_movies[:k]

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

class UserKNNRecommender:
    def __init__(self, k=5):
        self.k = k
        self.model = None
        self.user_movie_matrix = None
        self.user_ids = []

    def fit(self, df: pd.DataFrame):
        self.user_movie_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_ids = self.user_movie_matrix.index.tolist()
        self.model = NearestNeighbors(n_neighbors=self.k, algorithm='auto')
        self.model.fit(self.user_movie_matrix.values)

    def recommend(self, user_id: int, k=20):
        if user_id not in self.user_ids:
            return []

        user_idx = self.user_ids.index(user_id)
        distances, indices = self.model.kneighbors([self.user_movie_matrix.iloc[user_idx]], n_neighbors=self.k+1)
        neighbors = indices.flatten()[1:]  # omit self
        neighbor_ids = [self.user_ids[i] for i in neighbors]

        # Aggregate ratings from neighbors
        neighbor_ratings = self.user_movie_matrix.loc[neighbor_ids].mean().sort_values(ascending=False)
        already_watched = self.user_movie_matrix.columns[self.user_movie_matrix.loc[user_id] > 0]
        recommendations = [int(mid) for mid in neighbor_ratings.index if mid not in already_watched]

        return recommendations[:k]

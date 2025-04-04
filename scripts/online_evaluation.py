# scripts/online_evaluation.py

import pandas as pd
import pickle

# Cargar ratings y recomendaciones
ratings = pd.read_csv("data/ratings.csv")
with open("models/top_recommendations.pkl", "rb") as f:
    top_n = pickle.load(f)

# Definir un umbral para considerar una calificación como positiva
RATING_THRESHOLD = 4.0

# Evaluar Hit@10
hits = 0
total = 0

for user_id, group in ratings.groupby("userId"):
    relevant_movies = group[group["rating"] >= RATING_THRESHOLD]["movieId"].tolist()
    recommended_movies = top_n.get(user_id, [])

    if not recommended_movies:
        continue

    hit = any(movie in recommended_movies[:10] for movie in relevant_movies)
    hits += int(hit)
    total += 1

hit_at_10 = hits / total if total > 0 else 0

print(f"Hit@10: {hit_at_10:.4f}")

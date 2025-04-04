from flask import Flask
import pandas as pd
import pickle
import os

from surprise import SVD

app = Flask(__name__)

# Cargar datos
DATA_DIR = "../data/"
MODELS_DIR = "../models/"
ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

# Cargar modelos
with open(os.path.join(MODELS_DIR, "popularity_model.pkl"), "rb") as f:
    popularity_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "svd_model.pkl"), "rb") as f:
    svd_model: SVD = pickle.load(f)

# Películas disponibles
all_movie_ids = ratings["movieId"].unique()

@app.route("/recommend/<int:user_id>")
def recommend(user_id):
    try:
        # Predecir para cada película no evaluada por el usuario
        user_rated = ratings[ratings["userId"] == user_id]["movieId"].values
        unseen_movies = [mid for mid in all_movie_ids if mid not in user_rated]

        # Si hay pocas vistas, usar modelo de popularidad como fallback
        if len(user_rated) < 5:
            recommended_ids = popularity_model.index.tolist()
        else:
            predictions = [
                (movie_id, svd_model.predict(user_id, movie_id).est)
                for movie_id in unseen_movies
            ]
            top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:20]
            recommended_ids = [int(movie_id) for movie_id, _ in top_movies]

        # Formato requerido: movieId1,movieId2,...
        return ",".join(map(str, recommended_ids))

    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
